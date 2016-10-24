PyFactorizationMachines
=======================

An implementation of factorization machines, based on the model presented in *Factorization Machines* (Rendle 2010).

Dependencies
------------

PyFactorizationMachines requires Python 2.7 with numpy and theano.

The packages can be installed with pip.

    $ pip install numpy theano
    
Alternatively, these may be available through your operating system's package manager, or a package manager for your
Python environment (e.g., conda).

Additionally, scikit-learn is required for running example.py, which shows example usage of PyFactorizationMachines.

    $ pip install scikit-learn

How To Use
----------

To use PyFactorizationMachines, first import the *pyfm* module.

    >>> import pyfm
    
### Initializing a Model

A factorization machine is created with *FactorizationMachineClassifier* for a binary classification problem, or
FactorizationMachineRegressor for regression. Each constructor requires an argument specifying the number of features.

    >>> model = pyfm.FactorizationMachineRegressor(20)

FactorizationMachineRegressor and FactorizationMachineClassifier both take the following arguments:

* **feature_count** The dimensionality of each data point.
* **k** (optional; defaults to 8) Dimensionality of the factorization of pairwise interactions.
* **stdev** (optional; defaults to .01) The standard deviation of the normal distribution used to initialize the
interaction parameters of the model.
    
### Training a Model

A factorization machine is trained with the *fit* method, which takes as input a training dataset X, and a vector of
target values. For binary classification, the target values must be 1 or 0.

    >>> model.fit(X, y)

*fit* takes the following arguments:

* **X** An *n-by-d* numpy.ndarray with training data. The rows correspond to observations, and the columns correspond to
dimensions.
* **y** An numpy.ndarray vector with *n* target values corresponding to the *n* data points in *X*.
* **batch_size** (optional; defaults to 32) Number of samples per gradient update.
* **nb_epoch** (optional; defaults to 10)  The number of epochs to train the model.
* **shuffle** (optional; defaults to True) A flag indicating whether to shuffle the training samples at each epoch.
* **verbose** (optional; defaults to False) A flag specifying whether to log details to stdout when training the model.

### Predicting with a PyFactorizationMachine

For regression models, *FactorizationMachineRegressor.predict* can be used to predict the regression target for a
dataset X.

    >>> model.predict(X)

For binary classification models, *FactorizationMachineClassifier.predict* can be used to predict the class target for a
dataset X. *FactorizationMachineClassifier.predict_proba* returns the class 1 probability for a dataset X.

    >>> model.predict(X)
    >>> model.predict_proba(X)

*FactorizationMachineRegressor.predict*, *FactorizationMachineClassifier.predict*, and
*FactorizationMachineClassifier.predict_proba* take the following argument:

* **X** An *n-by-d* numpy.ndarray with data. The rows correspond to observations, and the columns correspond to
dimensions.

### Saving a PyFactorizationMachine

A factorization machine is saved with the *save* method, which takes as input a filename.

    >>> model.save("/path/to/save/model.mdl")

*save* takes the following arguments:

* **path** The file path for saving the model.

### Loading a Saved PyFactorizationMachine

A saved factorization machine can be loaded with the *load* top-level function.

    >>> fm = pyfm.load(""/path/to/model.mdl")

*load* takes the following arguments:

* **path** The file path for loading the saved model.

Differences from (Rendle 2010)
------------------------------

The paper suggests that stochastic gradient descent can be used for learning the parameters of the model. This
implementation specifically uses *Adam: A Method for Stochastic Optimization* (Kingma and Ba 2014).

For binary classification, this implementation uses a logit function combined with a cross entropy loss function.

License
-------

PyFactorizationMachines has an [MIT License](https://en.wikipedia.org/wiki/MIT_License).

See [LICENSE](LICENSE).

References
==========

Kingma, Diederik, and Jimmy Ba. 2014. “Adam: A Method for Stochastic Optimization.” arXiv:1412.6980 [Cs], December.
http://arxiv.org/abs/1412.6980.

Rendle, S. 2010. “Factorization Machines.” In 2010 IEEE 10th International Conference on Data Mining (ICDM), 995–1000.
doi:10.1109/ICDM.2010.127.
