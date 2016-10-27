"""
Abstract class for RGBD sensors.
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import numpy as np
import os
import random
import sys
import time

class RgbdSensor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def start(self):
        """ Starts the sensor stream """
        pass

    @abstractmethod
    def stop(self):
        """ Stops the sensor stream """
        pass

    def restart(self):
        """ Restarts the sensor stream """
        self.stop()
        self.start()

    @abstractmethod
    def frames(self):
        """ Returns the latest set of frames """
        pass

        
        
