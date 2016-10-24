from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pyfm

# This shows examples of how to use PyFactorizationMachines. The datasets may not be
# particularly suitable for using factorization machines.


def error_score(y_true, y_pred):
    return 1.0 - accuracy_score(y_true, y_pred)


print '*** Regression Example ***'

X, y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

factorization_machine_regressor = pyfm.FactorizationMachineRegressor(X.shape[1])
factorization_machine_regressor.fit(X_train, y_train, verbose=False, nb_epoch=5000)
print '  Factorization Machine MSE: {}'.format(
    mean_squared_error(y_test, factorization_machine_regressor.predict(X_test)))

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
print '  Linear Regression MSE: {}'.format(
    mean_squared_error(y_test, linear_regression.predict(X_test)))

print '\n*** Binary Classification Example ***'

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

factorization_machine_regressor = pyfm.FactorizationMachineClassifier(X.shape[1])
factorization_machine_regressor.fit(X_train, y_train, verbose=False, nb_epoch=5000)
print '  Factorization Machine Error: {}'.format(
    error_score(y_test, factorization_machine_regressor.predict(X_test)))

logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)
print '  Logistic Regression Error: {}'.format(
    error_score(y_test, logistic_regression.predict(X_test)))

# multi-class classification

# TODO: an example combining binary classifiers to make a multi-class factorization machine
# (e.g., one-versus-rest binary classifiers)

#X, y = datasets.load_iris(return_X_y=True)

print '\n*** Saving Model Example ***'

# Save the factorization machine regressor that was trained earlier

f = "fm.model"
factorization_machine_regressor.save(f)
print '  model saved'

print '\n*** Loading a Saved Model Example ***'

loaded_factorization_machine_regressor = pyfm.load(f)
print '  model loaded'
