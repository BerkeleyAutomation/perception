"""
Abstract class for Camera sensors.
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod


class CameraSensor(object):
    """Abstract base class for camera sensors."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def start(self):
        """Starts the sensor stream."""
        pass

    @abstractmethod
    def stop(self):
        """Stops the sensor stream."""
        pass

    def reset(self):
        """Restarts the sensor stream."""
        self.stop()
        self.start()

    @abstractmethod
    def frames(self):
        """Returns the latest set of frames."""
        pass
