import io
import os
from setuptools import setup

version_txt = os.path.join(os.path.dirname(__file__), 'pyfms', 'version.txt')
with open(version_txt, 'r') as f:
    version = f.read().strip()

setup(
  author = 'Daniel Steinberg',
  author_email = 'ds@dannyadam.com',
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'License :: OSI Approved :: MIT License',
    'Operating System :: Unix',
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6'
  ],
  description = 'A Theano-based Python implementation of Factorization Machines',
  extras_require={
    'dev': ['scikit-learn'],
  },
  install_requires = ['numpy', 'theano (>=0.8.0)'],
  keywords = ['factorization-machines', 'machine-learning'],
  license = 'MIT',
  long_description = io.open('README.rst', encoding='utf8').read(),
  name = 'pyfms',
  package_data={'pyfms': ['version.txt']},
  packages=['pyfms'],
  url = 'https://github.com/dstein64/pyfms',
  version = version,
)
