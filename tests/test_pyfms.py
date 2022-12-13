import os
import shutil
import tempfile
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
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_weighted_classifier(self):
        np.random.seed(0)
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Re-weight instances so that each class gets equal total weighting.
        class_count_lookup = dict(zip(*np.unique(y_train, return_counts=True)))
        sample_weight = np.array([1.0 / class_count_lookup[_y] for _y in y_train])

        classifier = pyfms.Classifier(X.shape[1])
        classifier.fit(X_train, y_train, sample_weight=sample_weight, nb_epoch=10000)

        accuracy = accuracy_score(y_test, classifier.predict(X_test))
        # Different accuracies are observed on different platforms.
        min_expected_accuracy = 0.9649122807017544
        self.assertGreaterEqual(accuracy, min_expected_accuracy)

    def test_sparse_classifier(self):
        np.random.seed(0)
        X, y = datasets.load_digits(return_X_y=True)
        X = sparse.csr_matrix(X)

        # pyfms only supports binary classifiers, so convert to a 0 versus other digits classifier.
        y = np.minimum(1, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        # Sparsify data
        X_train = sparse.csr_matrix(X_train)
        X_test = sparse.csr_matrix(X_test)

        classifier_dims = X.shape[1]
        classifier = pyfms.Classifier(classifier_dims, k=2, X_format="csr")
        classifier.fit(X_train, y_train, nb_epoch=2000)

        accuracy = accuracy_score(y_test, classifier.predict(X_test))
        self.assertAlmostEqual(accuracy, 0.9944382647385984)

    def test_regularized_regressor(self):
        np.random.seed(0)
        X, y = datasets.load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        regressor = pyfms.Regressor(X.shape[1], k=2)
        reg = pyfms.regularizers.L2(0, 0, .01)
        regressor.fit(X_train, y_train, nb_epoch=5000, regularizer=reg)

        mse = mean_squared_error(y_test, regressor.predict(X_test))
        self.assertAlmostEqual(mse, 23724.324079343063)

    def test_save_load_classifier(self):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        k = 4

        classifier_before = pyfms.Classifier(X.shape[1], k=k)
        classifier_before.fit(X_train, y_train, nb_epoch=1000)

        weights_before = classifier_before.get_weights()
        accuracy_before = accuracy_score(y_test, classifier_before.predict(X_test))

        classifier_file = os.path.join(self.workspace, 'classifier.fm')
        classifier_before.save_weights(classifier_file)

        classifier_after = pyfms.Classifier(X.shape[1])
        classifier_after.load_weights(classifier_file)

        weights_after = classifier_after.get_weights()
        accuracy_after = accuracy_score(y_test, classifier_after.predict(X_test))

        for wb, wa in zip(weights_before, weights_after):
            np.testing.assert_array_equal(wb, wa)
        self.assertEqual(accuracy_before, accuracy_after)

    def test_save_load_regressor(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        k = 4

        regressor_before = pyfms.Regressor(X.shape[1], k=k)
        regressor_before.fit(X_train, y_train, nb_epoch=1000)

        weights_before = regressor_before.get_weights()
        mse_before = mean_squared_error(y_test, regressor_before.predict(X_test))

        regressor_file = os.path.join(self.workspace, 'regressor.fm')
        regressor_before.save_weights(regressor_file)

        regressor_after = pyfms.Regressor(X.shape[1])
        regressor_after.load_weights(regressor_file)

        weights_after = regressor_after.get_weights()
        mse_after = mean_squared_error(y_test, regressor_after.predict(X_test))

        for wb, wa in zip(weights_before, weights_after):
            np.testing.assert_array_equal(wb, wa)
        self.assertEqual(mse_before, mse_after)


if __name__ == '__main__':
    unittest.main()
