import pyfm

import numpy as np

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def mse(actual, predicted):
    return np.mean((actual - predicted)**2)

def err(actual, predicted):
    return np.mean(actual != predicted)

# regression
X, y = datasets.load_boston(return_X_y=True)
fm = pyfm.FactorizationMachineRegressor(X.shape[1])
fm.fit(X, y, verbose=False, nb_epoch=5000)

f = "fm.model"
fm.save(f)
fm2 = pyfm.load(f)

print 'MSE: {}'.format(mse(y, fm.predict(X)))
print 'MSE: {}'.format(mse(y, fm2.predict(X)))

linear_regression = LinearRegression()
linear_regression.fit(X, y)
print 'MSE: {}'.format(mse(y, linear_regression.predict(X)))

# binary classification
X, y = datasets.load_breast_cancer(return_X_y=True)

fm = pyfm.FactorizationMachineClassifier(X.shape[1])
fm.fit(X, y, verbose=False, nb_epoch=5000)
print 'Error: {}'.format(err(y, fm.predict(X)))

logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)
print 'Error: {}'.format(err(y, logistic_regression.predict(X)))

# multi-class classification
#X, y = datasets.load_iris(return_X_y=True)
