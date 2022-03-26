"""
Setup of Berkeley AUTOLab Perception module Python codebase.
Author: Jeff Mahler
"""
import os

from setuptools import setup

requirements = [
    "numpy",
    "scipy",
    "autolab_core",
    "opencv-python",
    "pyserial>=3.4",
    "ffmpeg-python",
]

# load __version__ without importing anything
version_file = os.path.join(os.path.dirname(__file__), "perception/version.py")
with open(version_file, "r") as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split("=")[-1])

setup(
    name="autolab_perception",
    version=__version__,
    description="Perception utilities for the Berkeley AutoLab",
    author="Jeff Mahler",
    author_email="jmahler@berkeley.edu",
    maintainer="Mike Danielczuk",
    maintainer_email="mdanielczuk@berkeley.edu",
    license="Apache Software License",
    url="https://github.com/BerkeleyAutomation/perception",
    keywords="robotics grasping vision perception",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=["perception"],
    install_requires=requirements,
    extras_require={
        "docs": ["sphinx", "sphinxcontrib-napoleon", "sphinx_rtd_theme"],
        "ros": ["primesense", "rospkg", "catkin_pkg", "empy"],
    },
)
