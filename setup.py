"""
Setup of meshpy python codebase
Author: Jeff Mahler
"""
from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'autolab_core',
    'matplotlib',
    'multiprocess',
    'opencv-python',
    'keras',
    'cycler',
    'Pillow',
    'ipython',
    'scikit-image',
    'scikit-learn',
    'scikit-video'
]

exec(open('perception/version.py').read())

setup(name='autolab_perception',
      version=__version__,
      description='Perception utilities for the Berkeley AutoLab',
      author='Jeff Mahler',
      author_email='jmahler@berkeley.edu',
      license = 'Apache Software License',
      url = 'https://github.com/BerkeleyAutomation/perception',
      keywords = 'robotics grasping vision perception',
      classifiers = [
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering'
      ],
      packages=['perception'],
      setup_requres = requirements,
      install_requires = requirements,
      extras_require = { 'docs' : [
          'sphinx',
          'sphinxcontrib-napoleon',
          'sphinx_rtd_theme'
      ],
      }
)
