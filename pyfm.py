import numpy as np
import theano
from theano import tensor as T

class _FactorizationMachineBase:
    floatX = theano.config.floatX

    @staticmethod
    def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates

    def __init__(self):
        pass

    @staticmethod
    def _model(X, w0, w1, v):
        # the formula for pairwise interactions is from bottom left
        # of page 997 of Rendle 2010, "Factorization Machines."
        # This version scales linearly in k and d, as opposed to O(d^2).
        interactions = 0.5 * T.sum((T.dot(X, T.transpose(v)) ** 2) - T.dot(X ** 2, T.transpose(v ** 2)), axis=1)
        y_hat = T.addbroadcast(w0,0) + T.dot(X, w1) + interactions

        y_hat = T.nnet.sigmoid(y_hat)

        return y_hat


    def fit(self, X, y):
        """Build a factorization machine from the training set (X, y)."""

        k = 4

        n, d = X.shape

        # *** Symbolic variables ***
        X_sym = T.matrix()
        y_sym = T.vector()

        # *** Model parameters ***
        # bias term
        self.w0 = theano.shared(np.zeros(1))
        # first order coefficients
        self.w1 = theano.shared(np.zeros(d))
        # interaction factors
        self.v = theano.shared(np.random.random((k, d))*.01)

        y_hat = _FactorizationMachineBase._model(X_sym, self.w0, self.w1, self.v)

        #loss = T.mean((y_sym - y_hat)**2)
        loss = T.mean(T.nnet.binary_crossentropy(y_hat, y_sym))

        params = [self.w0, self.w1, self.v]
        updates = _FactorizationMachineBase.RMSprop(loss, params)

        train = theano.function(
            inputs=[X_sym, y_sym], outputs=loss, updates=updates, allow_input_downcast=True)
        self.theano_predict = theano.function(inputs=[X_sym], outputs=y_hat, allow_input_downcast=True)

        epochs = 1000
        for i in range(epochs):
            cost = train(X, y)
            print i, cost

    def predict(self, X):
        return self.theano_predict(X)

    def predict_proba(self, X):
        pass

from sklearn import datasets

# X, y = datasets.load_iris(return_X_y=True)

X, y = datasets.load_breast_cancer(return_X_y=True)

fm = _FactorizationMachineBase()
fm.fit(X, y)

print np.mean(y == (fm.predict(X) > 0.5).astype(np.int))

