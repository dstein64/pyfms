from distutils.core import setup

setup(
  name = 'pyfms',
  packages = ['pyfms'],
  license = 'MIT',
  version = '0.1.4',
  description = 'A Theano-based Python implementation of Factorization Machines',
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
    'Operating System :: Microsoft :: Windows',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 2.7'
  ],
  install_requires = ['numpy', 'theano>=0.8.0']
)
