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
print('* (with verbose output)')
print('*******************************************')
print()

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier_dims = X.shape[1]
fm_classifier = pyfms.Classifier(classifier_dims)
fm_classifier.fit(X_train, y_train, verbosity=2000, nb_epoch=10000)
print()
print('Factorization Machine Error: {}'.format(
    error_score(y_test, fm_classifier.predict(X_test))))

logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)
print('Logistic Regression Error: {}'.format(
    error_score(y_test, logistic_regression.predict(X_test))))
print()

print('*******************************************')
print('* Regression Example')
print('* (with sparse data and L2 Regularization)')
print('*******************************************')
print()

X, y = datasets.load_boston(return_X_y=True)
# Columns 1 and 3 (0-indexed) are sparse.
# Slice data to the first 5 columns for a higher sparsity ratio.
X = X[:,:5]
X = sparse.csr_matrix(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

fm_regressor = pyfms.Regressor(X.shape[1], k=2, X_format="csr")
reg = pyfms.regularizers.L2(0, 0, .01)
fm_regressor.fit(X_train, y_train, nb_epoch=30000, regularizer=reg)
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

# Save the factorization machine regressor that was trained earlier

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
