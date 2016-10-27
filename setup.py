"""
Setup of meshpy python codebase
Author: Jeff Mahler
"""
from setuptools import setup

setup(name='perception',
      version='0.1.dev0',
      description='Perception project code',
      author='Jeff Mahler',
      author_email='jmahler@berkeley.edu',
      package_dir = {'': '.'},
      packages=['perception'],
      test_suite='test'
     )
