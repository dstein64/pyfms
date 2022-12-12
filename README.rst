.. image:: https://github.com/dstein64/pyfms/workflows/build/badge.svg
    :target: https://github.com/dstein64/pyfms/actions

pyfms
=====

A `Theano <http://deeplearning.net/software/theano/>`__-based Python implementation of
factorization machines, based on the model presented in *Factorization Machines* (Rendle 2010).

Features
--------

-  Sample weighting
-  For binary classification, this implementation uses a logit function
   combined with a cross entropy loss function.
-  Extensibility of algorithms for: regularization, loss function optimization, and the error
   function
-  Support for sparse data

Requirements
------------

`pyfms` supports Python 3.x.

Linux and Mac are supported.

Windows is supported with Theano properly installed. The recommended way to install Theano on
Windows is using `Anaconda <https://www.continuum.io/anaconda-overview>`__.

::

    > conda install theano

Other operating systems may be compatible if Theano can be properly installed.

Installation
------------

`pyfms <https://pypi.python.org/pypi/pyfms>`__ is available on PyPI, the Python Package Index.

::

    $ pip install pyfms

Documentation
-------------

See `documentation.md <https://github.com/dstein64/pyfms/blob/master/documentation.md>`__.

Example Usage
-------------

See `example.py <https://github.com/dstein64/pyfms/blob/master/example.py>`__.

scikit-learn>=0.18 is required to run the example code.

Tests
-----

Tests are in `tests/ <https://github.com/dstein64/pyfms/blob/master/tests>`__.

::

    # Run tests
    $ python -m unittest discover tests -v

License
-------

`pyfms` has an `MIT License <https://en.wikipedia.org/wiki/MIT_License>`__.

See `LICENSE <https://github.com/dstein64/pyfms/blob/master/LICENSE>`__.

Acknowledgments
---------------

RMSprop code is from
`Newmu/Theano-Tutorials <https://github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py>`__.

Adam code is from
`Newmu/dcgan_code <https://github.com/Newmu/dcgan_code/blob/master/lib/updates.py>`__.

References
----------

Rendle, S. 2010. “Factorization Machines.” In 2010 IEEE 10th
International Conference on Data Mining (ICDM), 995–1000.
doi:10.1109/ICDM.2010.127.
