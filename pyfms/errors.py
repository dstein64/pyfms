from theano import tensor as T

import core

class SquaredError(core.Error):
    def apply(self, y, y_hat):
        return (y - y_hat)**2


class BinaryCrossEntropy(core.Error):
    def apply(self, y, y_hat):
        return T.nnet.binary_crossentropy(y_hat, y)
