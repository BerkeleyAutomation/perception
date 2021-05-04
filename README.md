## Berkeley Autolab Perception Module

[![Github Actions](https://github.com/BerkeleyAutomation/perception/actions/workflows/release.yml/badge.svg)](https://github.com/BerkeleyAutomation/perception/actions) [![PyPI version](https://badge.fury.io/py/autolab-perception.svg)](https://badge.fury.io/py/autolab-perception)

This package provides a wide variety of useful tools for perception tasks.
It directly depends on the [Berkeley Autolab Core
module](https://www.github.com/BerkeleyAutomation/autolab_core), so be sure to install
that first.
View the install guide and API documentation for the perception module
[here](https://BerkeleyAutomation.github.io/perception). Dependencies for each driver are not automatically installed, so please install ROS or camera-specific drivers separately before using these wrappers.

NOTE: As of May 4, 2021, this package no longer supports Python versions 3.5 or lower as these versions have reached EOL. In addition, many modules have been moved to `autolab_core` to reduce confusion. This repository now will contain sensor drivers and interfaces only. If you wish to use older Python versions or rely on the old modules, please use the 0.x.x series of tags.