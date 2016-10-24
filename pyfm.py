import numpy as np
import theano
from theano import tensor as T

class FactorizationMachine(object):
    def __init__(self,
                 feature_count,
                 classifier=False,
                 k = 8,
                 stdev = 0.1):
        self.classifier = classifier
        d = feature_count

        # *** Symbolic variables ***
        X = T.matrix()
        y = T.vector()

        # *** Model parameters ***
        # bias term
        w0_init = np.zeros(1)
        self.w0 = theano.shared(w0_init, allow_downcast=True)
        # first order coefficients
        w1_init = np.zeros(d)
        self.w1 = theano.shared(w1_init, allow_downcast=True)
        # interaction factors
        v_init = stdev * np.random.randn(k, d)
        self.v = theano.shared(v_init, allow_downcast=True)

        # *** The Model ***
        # the formula for pairwise interactions is from bottom left
        # of page 997 of Rendle 2010, "Factorization Machines."
        # This version scales linearly in k and d, as opposed to O(d^2).
        interactions = 0.5 * T.sum((T.dot(X, T.transpose(self.v)) ** 2) - \
                                   T.dot(X ** 2, T.transpose(self.v ** 2)), axis=1)
        y_hat = T.addbroadcast(self.w0,0) + T.dot(X, self.w1) + interactions
        if self.classifier:
            y_hat = T.nnet.sigmoid(y_hat)

        # *** Loss Function ***
        if self.classifier:
            loss = T.mean(T.nnet.binary_crossentropy(y_hat, y))
        else:
            loss = T.mean((y - y_hat)**2)

        # *** Learning ***
        updates = []
        params = [self.w0, self.w1, self.v]
        grads = T.grad(cost=loss, wrt=params)
        # rmsprop
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

        # *** Prediction ***
        self.theano_predict = theano.function(
            inputs=[X], outputs=y_hat, allow_input_downcast=True)

    def fit(self, X, y, epochs=100000, verbose=False):
        for i in range(epochs):
            cost = self.theano_train(X, y)
            if verbose:
                print i, cost

    def save(self, path):
        with open(path, 'wb') as f:
            np.savez(f,
                     classifier=self.classifier,
                     w0=self.w0.get_value(),
                     w1=self.w1.get_value(),
                     v=self.v.get_value())

class FactorizationMachineClassifier(FactorizationMachine, object):
    def __init__(self, *args, **kwargs):
        kwargs['classifier'] = True
        super(FactorizationMachineClassifier, self).__init__(*args, **kwargs)

    def predict(self, X):
        return (self.theano_predict(X) > 0.5).astype(np.int)

    def predict_proba(self, X):
        return self.theano_predict(X)

class FactorizationMachineRegressor(FactorizationMachine, object):
    def __init__(self, *args, **kwargs):
        kwargs['classifier'] = False
        super(FactorizationMachineRegressor, self).__init__(*args, **kwargs)

    def predict(self, X):
        return self.theano_predict(X)

def load(path):
    meta = np.load(path)
    classifier = meta['classifier']
    w0, w1, v = [meta[key] for key in ['w0', 'w1', 'v']]
    k, d = v.shape
    cls = FactorizationMachineClassifier if classifier else FactorizationMachineRegressor
    model = cls(d, k=k)
    model.w0.set_value(w0)
    model.w1.set_value(w1)
    model.v.set_value(v)
    return model
