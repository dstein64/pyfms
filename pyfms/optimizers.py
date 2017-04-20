import theano
from theano import tensor as T

import core

class RMSProp(core.Optimizer):
    def __init__(self,
                 lr = 0.001,
                 rho = 0.9,
                 epsilon = 1e-6):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

    def update(self, loss, params, epoch):
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


class SGD(core.Optimizer):
    def __init__(self, lr = 0.001, decay = 0.0):
        self.lr = lr
        self.decay = decay

    def update(self, loss, params, epoch):
        updates = []
        grads = T.grad(cost=loss, wrt=params)
        for p, g in zip(params, grads):
            lr = self.lr * (1 / (1 + self.decay * (epoch-1)))
            updates.append((p, p - lr * g))
        return updates
