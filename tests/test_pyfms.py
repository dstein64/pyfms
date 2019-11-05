import unittest

import numpy as np
from scipy import sparse
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pyfms
import pyfms.regularizers

class TestPyfms(unittest.TestCase):
    def test_weighted_classifier(self):
        np.random.seed(0)
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Re-weight instances so that each class gets equal total weighting.
        class_count_lookup = dict(zip(*np.unique(y_train, return_counts=True)))
        sample_weight = np.array([1.0 / class_count_lookup[_y] for _y in y_train])

        fm_classifier = pyfms.Classifier(X.shape[1])
        fm_classifier.fit(X_train, y_train, sample_weight=sample_weight, nb_epoch=10000)

        accuracy = accuracy_score(y_test, fm_classifier.predict(X_test))
        self.assertAlmostEqual(accuracy, 0.9649122807017544)

    def test_sparse_classifier(self):
        np.random.seed(0)
        X, y = datasets.load_boston(return_X_y=True)
        y = y > 30 # Binarize target
        # Columns 1 and 3 (0-indexed) are sparse.
        # Slice data to the first 5 columns for a higher sparsity ratio.
        X = X[:, :5]
        X = sparse.csr_matrix(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Sparsify data
        X_train = sparse.csr_matrix(X_train)
        X_test = sparse.csr_matrix(X_test)

        classifier_dims = X.shape[1]
        fm_classifier = pyfms.Classifier(classifier_dims, k=2, X_format="csr")
        fm_classifier.fit(X_train, y_train, nb_epoch=20000)

        accuracy = accuracy_score(y_test, fm_classifier.predict(X_test))
        self.assertAlmostEqual(accuracy, 0.8725490196078431)

    def test_regularized_regressor(self):
        np.random.seed(0)
        X, y = datasets.load_boston(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        fm_regressor = pyfms.Regressor(X.shape[1], k=2)
        reg = pyfms.regularizers.L2(0, 0, .01)
        fm_regressor.fit(X_train, y_train, nb_epoch=50000, regularizer=reg)

        mse = mean_squared_error(y_test, fm_regressor.predict(X_test))
        self.assertAlmostEqual(mse, 26.28317095306481)


if __name__ == '__main__':
    unittest.main()
