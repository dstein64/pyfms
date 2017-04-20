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


class Adam(core.Optimizer):
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, l=1-1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.l = l

    def update(self, loss, params, epoch):
        updates = []
        grads = T.grad(loss, params)
        t = theano.shared(1., allow_downcast=True)
        b1_t = self.b1 * self.l ** (t - 1)

        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = b1_t * m + (1 - b1_t) * g
            v_t = self.b2 * v + (1 - self.b2) * g ** 2
            m_c = m_t / (1 - self.b1 ** t)
            v_c = v_t / (1 - self.b2 ** t)
            p_t = p - (self.lr * m_c) / (T.sqrt(v_c) + self.e)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((t, t + 1.))
        return updates
