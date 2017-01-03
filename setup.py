"""
Setup of meshpy python codebase
Author: Jeff Mahler
"""
from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'opencv-python',
    'cycler',
    'tensorflow',
    'Pillow',
    'core'
]

dependency_links=['https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl']

setup(name='perception',
      version='0.1.dev0',
      description='Perception project code',
      author='Jeff Mahler',
      author_email='jmahler@berkeley.edu',
      package_dir = {'': '.'},
      packages=['perception'],
      install_requires=requirements,
      dependency_links=dependency_links,
      test_suite='test'
     )
