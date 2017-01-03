"""
Abstract class for RGBD sensors.
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

class RgbdSensor(object):
    """Abstract base class for red-green-blue-depth sensors.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def start(self):
        """Starts the sensor stream.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stops the sensor stream.
        """
        pass

    def reset(self):
        """Restarts the sensor stream.
        """
        self.stop()
        self.start()

    @abstractmethod
    def frames(self):
        """Returns the latest set of frames.
        """
        pass

