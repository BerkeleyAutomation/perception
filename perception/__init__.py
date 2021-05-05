"""
Autolab Drivers
Authors: Jeff, Jacky
"""
from .camera_sensor import CameraSensor
from .rgbd_sensors import RgbdSensorFactory

try:
    from .weight_sensor import WeightSensor
except BaseException as E:
    from . import exceptions

    WeightSensor = exceptions.closure(E)

from .video_recorder import VideoRecorder
