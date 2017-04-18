import abc
import itertools
from collections import namedtuple

import numpy as np
import theano
from theano import tensor as T

_Weights = namedtuple('_Weights', ['w0', 'w1', 'v'])


class _Optimizer(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def update(self, cost, params):
        raise NotImplementedError()


class RMSProp(_Optimizer):
    def __init__(self,
                 lr = 0.001,
                 rho = 0.9,
                 epsilon = 1e-6):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

    def update(self, loss, params):
        updates = []
        grads = T.grad(cost=loss, wrt=params)
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0., allow_downcast=True)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + self.epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - self.lr * g))
        return updates


class SGD(_Optimizer):
    def __init__(self, lr = 0.001):
        self.lr = lr

    def updates(self, loss, params):
        updates = []
        grads = T.grad(cost=loss, wrt=params)
        for p, g in zip(params, grads):
            updates.append((p, p - self.lr * g))
        return updates


class _ErrorFunction(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def apply(self, y, y_hat):
        raise NotImplementedError()


class SquaredError(_ErrorFunction):
    def apply(self, y, y_hat):
        return (y - y_hat)**2


class BinaryCrossEntropy(_ErrorFunction):
    def apply(self, y, y_hat):
        return T.nnet.binary_crossentropy(y_hat, y)


class _Regularizer(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def regularize(self, loss, w0, w1, v):
        raise NotImplementedError()


class L2(_Regularizer):
    def __init__(self, beta_w0 = 0.0, beta_w1 = 0.0, beta_v = 0.0):
        self.beta_w0 = beta_w0
        self.beta_w1 = beta_w1
        self.beta_v = beta_v

    def regularize(self, loss, w0, w1, v):
        penalty = (self.beta_w0 * w0) \
                  + (self.beta_w1 * T.mean(w1 ** 2)) \
                  + (self.beta_v * T.mean(v ** 2))
        return loss + penalty


# This is used for transforming PyFactorizationMachine's output
class _Transformer(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def transform(self, y_hat):
        raise NotImplementedError()


class Linear(_Transformer):
    def transform(self, y_hat):
        return y_hat


class Sigmoid(_Transformer):
    def transform(self, y_hat):
        return T.nnet.sigmoid(y_hat)


class _FactorizationMachine(object):
    """Base class for factorization machines.

    Warning: This class should not be used directly. Use derived classes
    instead (FactorizationMachineClassifier and FactorizationMachineRegressor).
    """
    def __init__(self,
                 feature_count,
                 transformer,
                 k = 8,
                 stdev = 0.1):
        d = feature_count

        # ************************************************************
        # * Symbolic Variables
        # ************************************************************

        self.X = T.matrix() # design matrix
        self.y = T.vector() # response
        self.s = T.vector() # sample weights

        # ************************************************************
        # * Model Parameters
        # ************************************************************

        # bias term (intercept)
        w0_init = np.zeros(1)
        self.w0 = theano.shared(w0_init, allow_downcast=True)
        # first order coefficients
        w1_init = np.zeros(d)
        self.w1 = theano.shared(w1_init, allow_downcast=True)
        # interaction factors
        v_init = stdev * np.random.randn(k, d)
        self.v = theano.shared(v_init, allow_downcast=True)

        # ************************************************************
        # * The Model
        # ************************************************************

        # The formula for pairwise interactions is from the bottom left
        # of page 997 of Rendle 2010, "Factorization Machines."
        # This version scales linearly in k and d, as opposed to O(d^2).
        interactions = 0.5 * T.sum((T.dot(self.X, T.transpose(self.v)) ** 2) \
                                   - T.dot(self.X ** 2, T.transpose(self.v ** 2)), axis=1)
        self.y_hat = self.w0[0] + T.dot(self.X, self.w1) + interactions
        self.y_hat = transformer.transform(self.y_hat)

        # ************************************************************
        # * Prediction
        # ************************************************************

        self.theano_predict = theano.function(
            inputs=[self.X], outputs=self.y_hat, allow_input_downcast=True)


    def get_weights(self):
        """Returns a _Weights namedtuple"""
        return _Weights(*(w.get_value() for w in (self.w0, self.w1, self.v)))


    def set_weights(self, weights):
        """Sets weights from a _Weights namedtuple"""
        self.w0.set_value(weights.w0)
        self.w1.set_value(weights.w1)
        self.v.set_value(weights.v)


    def fit(self, X_train, y_train,
            error_function,
            optimizer = RMSProp(),
            regularizer = None,
            sample_weight = None,
            batch_size = 128,
            nb_epoch = 100,
            shuffle = True,
            verbose = False,
            memory = True):
        """Learns the weights of a factorization machine with mini-batch gradient
        descent. The weights that minimize the loss function (across epochs) are
        retained."""

        # ************************************************************
        # * Learning (Symbolic)
        # ************************************************************

        # *** Loss Function ***
        error = error_function.apply(self.y, self.y_hat)
        mean_error = T.true_div(T.sum(T.mul(error, self.s)), T.sum(self.s))
        loss = mean_error
        # regularization
        if regularizer is not None:
            loss = regularizer.regularize(loss, self.w0[0], self.w1, self.v)

        params = [self.w0, self.w1, self.v]
        updates = optimizer.update(loss, params)

        theano_train = theano.function(
            inputs=[self.X, self.y, self.s], outputs=loss, updates=updates, allow_input_downcast=True)

        theano_cost = theano.function(
            inputs=[self.X, self.y, self.s], outputs=loss, allow_input_downcast=True)

        # ************************************************************
        # * Learning (Numeric)
        # ************************************************************

        n = X_train.shape[0]
        if batch_size > n:
            batch_size = n
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            sample_weight = (float(n) / np.sum(sample_weight)) * sample_weight
        min_loss = float('inf')
        min_loss_weights = self.get_weights()
        for i in range(nb_epoch):
            if shuffle:
                indices = np.arange(n)
                np.random.shuffle(indices)
                X_train, y_train = X_train[indices], y_train[indices]
            for start in itertools.count(0, batch_size):
                if start >= n:
                    break
                stop = min(start + batch_size, n)
                theano_train(X_train[start:stop],
                             y_train[start:stop],
                             sample_weight[start:stop])
            current_loss = theano_cost(X_train, y_train, sample_weight)
            if not np.isfinite(current_loss):
                raise ArithmeticError()
            if current_loss < min_loss:
                min_loss = current_loss
                min_loss_weights = self.get_weights()
            if verbose:
                print 'Epoch {}/{}'.format(i+1, nb_epoch)
                print ' loss: {}, min_loss: {}'.format(current_loss, min_loss)
        weights = min_loss_weights if memory else self.get_weights()
        self.set_weights(weights)


    def save_weights(self, path):
        with open(path, 'wb') as f:
            w0, w1, v = self.get_weights()
            np.savez(
                f, w0=w0, w1=w1, v=v)


    def load_weights(self, path):
        meta = np.load(path)
        weights = _Weights(*[meta[key] for key in ['w0', 'w1', 'v']])
        self.set_weights(weights)


class FactorizationMachineClassifier(_FactorizationMachine, object):
    """A factorization machine classifier."""
    def __init__(self, feature_count, **kwargs):
        transformer = Sigmoid()
        super(FactorizationMachineClassifier, self).__init__(
            feature_count, transformer, **kwargs)


    def fit(self, X_train, y_train, **kwargs):
        error_function = BinaryCrossEntropy()
        super(FactorizationMachineClassifier, self).fit(
            X_train, y_train, error_function, **kwargs)


    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.int)


    def predict_proba(self, X):
        return self.theano_predict(X)


class FactorizationMachineRegressor(_FactorizationMachine, object):
    """A factorization machine regressor."""
    def __init__(self, feature_count, **kwargs):
        transformer = Linear()
        super(FactorizationMachineRegressor, self).__init__(
            feature_count, transformer, **kwargs)


    def fit(self, X_train, y_train, **kwargs):
        error_function = SquaredError()
        super(FactorizationMachineRegressor, self).fit(
            X_train, y_train, error_function, **kwargs)


    def predict(self, X):
        return self.theano_predict(X)
