from __future__ import print_function

import numpy as np
from scipy import sparse
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pyfms
import pyfms.regularizers

# This shows examples of how to use PyFactorizationMachines. The datasets may not be
# particularly suitable for using factorization machines.

print('pyfms {}'.format(pyfms.__version__))
print()

np.random.seed(0)

def error_score(y_true, y_pred):
    return 1.0 - accuracy_score(y_true, y_pred)

print('*******************************************')
print('* Binary Classification Example')
print('* (with sample weighting and sparse data)')
print('*******************************************')
print()

X, y = datasets.load_boston(return_X_y=True)

# Binarize target
y = y > 30

# Columns 1 and 3 (0-indexed) are sparse.
# Slice data to the first 5 columns for a higher sparsity ratio.
X = X[:,:5]
X = sparse.csr_matrix(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Re-weight instances so that each class gets equal total weighting.
class_count_lookup = dict(zip(*np.unique(y_train, return_counts=True)))
sample_weight = np.array([1.0 / class_count_lookup[_y] for _y in y_train])

# Sparsify data
X_train = sparse.csr_matrix(X_train)
X_test = sparse.csr_matrix(X_test)

classifier_dims = X.shape[1]
fm_classifier = pyfms.Classifier(classifier_dims, k=2, X_format="csr")
fm_classifier.fit(X_train, y_train, sample_weight=sample_weight, nb_epoch=20000)
print('Factorization Machine Error: {}'.format(
    error_score(y_test, fm_classifier.predict(X_test))))

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train, sample_weight=sample_weight)
print('Logistic Regression Error: {}'.format(
    error_score(y_test, logistic_regression.predict(X_test))))
print()

print('*******************************************')
print('* Regression Example')
print('* (with L2 Regularization and verbose output)')
print('*******************************************')
print()

X, y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

fm_regressor = pyfms.Regressor(X.shape[1], k=2)
reg = pyfms.regularizers.L2(0, 0, .01)
fm_regressor.fit(X_train, y_train, nb_epoch=50000, verbosity=5000, regularizer=reg)
print()
print('Factorization Machine MSE: {}'.format(
    mean_squared_error(y_test, fm_regressor.predict(X_test))))

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
print('Linear Regression MSE: {}'.format(
    mean_squared_error(y_test, linear_regression.predict(X_test))))
print()

print('*******************************************')
print('* Saving Model Example')
print('*******************************************')
print()

# Save the factorization machine classifier that was trained earlier

f = "weights.fm"
fm_classifier.save_weights(f)
print('Model saved')
print()

print('*******************************************')
print('* Loading a Saved Model Example')
print('*******************************************')
print()

del fm_classifier

fm_classifier = pyfms.models.Classifier(classifier_dims)
fm_classifier.load_weights(f)
print('Model loaded')
print()
