"""
Class for interfacing with the Kinect v2 RGBD sensor
Author: Jeff Mahler
"""
import copy
import logging
import numpy as np
import os

from constants import MM_TO_METERS, INTR_EXTENSION
try:
    from primesense import openni2
except:
    logging.warning('Unable to import openni2 driver. Primesense sensor may not work properly')

from camera_intrinsics import CameraIntrinsics
from image import ColorImage, DepthImage, IrImage, Image
from rgbd_sensor import RgbdSensor

class PrimesenseSensor(RgbdSensor):
    """ Class for interacting with a Primesense RGBD sensor.
    """
    #Constants for image height and width (in case they're needed somewhere)
    COLOR_IM_HEIGHT = 480
    COLOR_IM_WIDTH = 640
    DEPTH_IM_HEIGHT = 480
    DEPTH_IM_WIDTH = 640
    CENTER_X = DEPTH_IM_WIDTH / 2.0
    CENTER_Y = DEPTH_IM_HEIGHT / 2.0
    FOCAL_X = 525.
    FOCAL_Y = 525.
    FPS = 30
    OPENNI2_PATH = '/home/jmahler/Libraries/OpenNI-Linux-x64-2.2/Redist'

    def __init__(self, device_num):
        self._device = None
        self._depth_stream = None
        self._color_stream = None
        self._running = None

        self._device_num = device_num
        self._frame = frame

        if self._frame is None:
            self._frame = 'primesense_%d' %(self._device_num)
        self._color_frame = '%s_color' %(self._frame)
        self._ir_frame = self._frame # same as color since we normally use this one

        openni2.initialize(PrimesenseSensor.OPENNI2_PATH)

    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the primesense IR camera.
        """
        return CameraIntrinsics(self._ir_frame, PrimesenseSensor.FOCAL_X, PrimesenseSensor.FOCAL_Y,
                                PrimesenseSensor.CENTER_X, PrimesenseSensor.CENTER_Y,
                                height=PrimesenseSensor.DEPTH_IM_HEIGHT,
                                width=PrimesenseSensor.DEPTH_IM_WIDTH)

    
