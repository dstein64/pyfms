import os
from setuptools import setup

version_txt = os.path.join(os.path.dirname(__file__), 'pyfms', 'version.txt')
with open(version_txt, 'r') as f:
    version = f.read().strip()

setup(
  name = 'pyfms',
  packages = ['pyfms'],
  package_data = {'pyfms': ['version.txt']},
  license = 'MIT',
  version = version,
  description = 'A Theano-based Python implementation of Factorization Machines',
  long_description = open('README.rst').read(),
  author = 'Daniel Steinberg',
  author_email = 'ds@dannyadam.com',
  url = 'https://github.com/dstein64/PyFactorizationMachines',
  keywords = ['factorization-machines', 'machine-learning'],
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
  install_requires = ['numpy', 'theano (>=0.8.0)']
)
