import numpy as np

from . import core, errors, optimizers, transformers

class Classifier(core.Model):
    """A factorization machine classifier."""
    def __init__(self, feature_count, **kwargs):
        transformer = transformers.Sigmoid()
        super(Classifier, self).__init__(
            feature_count, transformer, **kwargs)


    def fit(self,
            X_train,
            y_train,
            optimizer=optimizers.Adam(),
            **kwargs):
        error_function = errors.BinaryCrossEntropy()
        super(Classifier, self).fit(
            X_train, y_train, error_function, optimizer, **kwargs)


    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.int)


    def predict_proba(self, X):
        return self.theano_predict(X)


class Regressor(core.Model):
    """A factorization machine regressor."""
    def __init__(self, feature_count, **kwargs):
        transformer = transformers.Linear()
        super(Regressor, self).__init__(
            feature_count, transformer, **kwargs)


    def fit(self,
            X_train,
            y_train,
            optimizer=optimizers.Adam(),
            **kwargs):
        error_function = errors.SquaredError()
        super(Regressor, self).fit(
            X_train, y_train, error_function, optimizer, **kwargs)


    def predict(self, X):
        return self.theano_predict(X)
