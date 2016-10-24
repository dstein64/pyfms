import itertools
from collections import namedtuple

import numpy as np
import theano
from theano import tensor as T


_Weights = namedtuple('_Weights', ['w0', 'w1', 'v'])


class _FactorizationMachine(object):
    """Base class for factorization machines.

    Warning: This class should not be used directly. Use derived classes
    instead (FactorizationMachineClassifier and FactorizationMachineRegressor).
    """
    def __init__(self,
                 feature_count,
                 classifier=False,
                 k = 8,
                 stdev = 0.1,
                 beta_w1 = 0.0,
                 beta_v = 0.0):
        self.classifier = classifier
        d = feature_count

        # *** Symbolic variables ***
        X = T.matrix()
        y = T.vector()

        # *** Model parameters ***
        # bias term (intercept)
        w0_init = np.zeros(1)
        self.w0 = theano.shared(w0_init, allow_downcast=True)
        # first order coefficients
        w1_init = np.zeros(d)
        self.w1 = theano.shared(w1_init, allow_downcast=True)
        # interaction factors
        v_init = stdev * np.random.randn(k, d)
        self.v = theano.shared(v_init, allow_downcast=True)

        # *** The Model ***
        # The formula for pairwise interactions is from the bottom left
        # of page 997 of Rendle 2010, "Factorization Machines."
        # This version scales linearly in k and d, as opposed to O(d^2).
        interactions = 0.5 * T.sum((T.dot(X, T.transpose(self.v)) ** 2) - \
                                   T.dot(X ** 2, T.transpose(self.v ** 2)), axis=1)
        y_hat = T.addbroadcast(self.w0,0) + T.dot(X, self.w1) + interactions
        if self.classifier:
            y_hat = T.nnet.sigmoid(y_hat)

        # *** Loss Function ***
        if self.classifier:
            error = T.mean(T.nnet.binary_crossentropy(y_hat, y))
        else:
            error = T.mean((y - y_hat)**2)
        # regularization
        L2 = 0.0
        if beta_w1 > 0.0 or beta_v > 0.0:
            L2 = beta_w1 * T.mean(self.w1 ** 2) + beta_v * T.mean(self.v ** 2)
        loss = error + L2

        # *** Learning ***
        updates = []
        params = [self.w0, self.w1, self.v]
        grads = T.grad(cost=loss, wrt=params)
        # rmsprop TODO: adam
        lr, rho, epsilon = 0.001, 0.9, 1e-6
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))

        self.theano_train = theano.function(
            inputs=[X, y], outputs=loss, updates=updates, allow_input_downcast=True)

        self.theano_cost = theano.function(
            inputs=[X, y], outputs=loss, allow_input_downcast=True)

        # *** Prediction ***
        self.theano_predict = theano.function(
            inputs=[X], outputs=y_hat, allow_input_downcast=True)


    def get_weights(self):
        """Returns a _Weights namedtuple"""
        return _Weights(*(w.get_value() for w in (self.w0, self.w1, self.v)))


    def set_weights(self, weights):
        """Sets weights from a _Weights namedtuple"""
        self.w0.set_value(weights.w0)
        self.w1.set_value(weights.w1)
        self.v.set_value(weights.v)


    def fit(self, X, y, batch_size=32, nb_epoch=10, shuffle=True, verbose=False):
        """Learns the weights of a factorization machine with mini-batch gradient
        descent. The weights that minimize the loss function (across epochs) are
        retained."""
        # TODO: support for validation
        n = X.shape[0]
        if batch_size > n:
            batch_size = n
        min_loss = float('inf')
        min_loss_weights = self.get_weights()
        for i in range(nb_epoch):
            if shuffle:
                indices = np.arange(n)
                np.random.shuffle(indices)
                X, y = X[indices], y[indices]
            for start in itertools.count(0, batch_size):
                if start >= n:
                    break
                stop = min(start + batch_size, n)
                self.theano_train(X[start:stop], y[start:stop])
            current_loss = self.theano_cost(X, y)
            if current_loss < min_loss:
                min_loss = current_loss
                min_loss_weights = self.get_weights()
            if verbose:
                print 'Epoch {}/{}'.format(i+1, nb_epoch)
                print ' loss: {}, min_loss: {}'.format(current_loss, min_loss)
        self.set_weights(min_loss_weights)


    def save(self, path):
        with open(path, 'wb') as f:
            w0, w1, v = self.get_weights()
            np.savez(
                f, classifier=self.classifier, w0=w0, w1=w1, v=v)


class FactorizationMachineClassifier(_FactorizationMachine, object):
    """A factorization machine classifier."""
    def __init__(self, *args, **kwargs):
        kwargs['classifier'] = True
        super(FactorizationMachineClassifier, self).__init__(*args, **kwargs)


    def predict(self, X):
        return (self.theano_predict(X) > 0.5).astype(np.int)


    def predict_proba(self, X):
        return self.theano_predict(X)


class FactorizationMachineRegressor(_FactorizationMachine, object):
    """A factorization machine regressor."""
    def __init__(self, *args, **kwargs):
        kwargs['classifier'] = False
        super(FactorizationMachineRegressor, self).__init__(*args, **kwargs)


    def predict(self, X):
        return self.theano_predict(X)


def load(path):
    meta = np.load(path)
    classifier = meta['classifier']
    weights = _Weights(*[meta[key] for key in ['w0', 'w1', 'v']])
    k, d = weights.v.shape
    cls = FactorizationMachineClassifier if classifier else FactorizationMachineRegressor
    model = cls(d, k=k)
    model.set_weights(weights)
    return model
