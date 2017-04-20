import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pyfms.models
import pyfms.regularizers

# This shows examples of how to use PyFactorizationMachines. The datasets may not be
# particularly suitable for using factorization machines.

np.random.seed(0)


def error_score(y_true, y_pred):
    return 1.0 - accuracy_score(y_true, y_pred)


print '*** Regression Example (with L2 Regularization) ***'

X, y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

fm_regressor = pyfms.models.Regressor(X.shape[1], k=2)
reg = pyfms.regularizers.L2(0, 0, 10)
fm_regressor.fit(X_train, y_train, nb_epoch=20000, regularizer=reg)
print '  Factorization Machine MSE: {}'.format(
    mean_squared_error(y_test, fm_regressor.predict(X_test)))

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
print '  Linear Regression MSE: {}'.format(
    mean_squared_error(y_test, linear_regression.predict(X_test)))

print '\n*** Binary Classification Example (with verbose output) ***'

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

fm_classifier = pyfms.models.Classifier(X.shape[1])
fm_classifier.fit(X_train, y_train, verbosity=1000, nb_epoch=5000)
print '  Factorization Machine Error: {}'.format(
    error_score(y_test, fm_classifier.predict(X_test)))

logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)
print '  Logistic Regression Error: {}'.format(
    error_score(y_test, logistic_regression.predict(X_test)))

print '\n*** Saving Model Example ***'

# Save the factorization machine classifier that was trained earlier

f = "weights.fm"
fm_classifier.save_weights(f)
print '  model saved'

print '\n*** Loading a Saved Model Example ***'

del fm_classifier

fm_classifier = pyfms.models.Classifier(X.shape[1])
fm_classifier.load_weights(f)
print '  model loaded'
