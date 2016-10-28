'''
Abstraction for interacting with video devices that interface with OpenCV
Author: Jacky Liang
'''

from camera_sensor import CameraSensor
from image import ColorImage
from cv2 import VideoCapture
from time import time
class OpenCVCameraSensor(CameraSensor):

    def __init__(self, device_id):
        self._device_id = device_id

    def start(self):
        """ Starts the OpenCVCameraSensor Stream
        Raises:
            Exception if unable to open stream
        """
        self._sensor = VideoCapture(self._device_id)
        if not self._sensor.isOpened():
            raise Exception("Unable to open OpenCVCameraSensor for id {0}".format(self._device_id))
        for _ in range(4):
            self._sensor.read()

    def stop(self):
        """ Stops the OpenCVCameraSensor Stream """
        self._sensor.release()

    def frames(self):
        """ Returns the latest color image from the stream
        Raises:
            Exception if opencv sensor gives ret_val of 0
        """
        ret_val, frame = self._sensor.read()
        if not ret_val:
            raise Exception("Unable to retrieve frame from OpenCVCameraSensor for id {0}".format(self._device_id))
        return ColorImage(frame)
