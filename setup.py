"""
Setup of meshpy python codebase
Author: Jeff Mahler
"""
from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'sklearn',
    'cycler',
    'Pillow',
    'ipython',
    'scikit-image'
]

setup(name='perception',
      version='0.1.0',
      description='Perception project code',
      author='Jeff Mahler',
      author_email='jmahler@berkeley.edu',
      package_dir = {'': '.'},
      packages=['perception'],
      install_requires=requirements,
      test_suite='test'
     )
