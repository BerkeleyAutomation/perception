import logging
import os
import subprocess

import cv2
from autolab_core import CameraIntrinsics, ColorImage

from .camera_sensor import CameraSensor


class WebcamSensor(CameraSensor):
    """Class for interfacing with a Logitech webcam sensor."""

    def __init__(self, frame="webcam", intrinsics=None, device_id=0):
        """Initialize a Logitech webcam sensor.

        Parameters
        ----------
        intrinsics : CameraIntrinsics
            Camera intrinsics object for the camera (can be found using calibrate_camera.py)
        device_id : int
            The device ID for the webcam (by default, zero).
        """
        self._frame = frame
        self._camera_intr = CameraIntrinsics.load(intrinsics) if intrinsics is not None else None
        self._device_id = device_id
        self._cap = None
        self._running = False
        self._adjust_exposure = True

    def __del__(self):
        """Automatically stop the sensor for safety."""
        if self.is_running:
            self.stop()

    @property
    def color_intrinsics(self):
        """Camera intrinsics for the webcam."""
        return self._camera_intr

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise."""
        return self._running

    @property
    def frame(self):
        """str : The reference frame of the sensor."""
        return self._frame

    @property
    def color_frame(self):
        """str : The reference frame of the sensor."""
        return self._frame

    def start(self):
        """Start the sensor."""
        self._cap = cv2.VideoCapture(self._device_id + cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._running = False
            self._cap.release()
            self._cap = None
            return False

        if self._camera_intr is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_intr.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_intr.height)
        self._running = True

        # Capture 5 frames to flush webcam sensor
        for _ in range(5):
            _ = self.frames()

        return True

    def stop(self):
        """Stop the sensor."""
        # Check that everything is running
        if not self._running:
            logging.warning("Webcam not running. Aborting stop")
            return False

        if self._cap:
            self._cap.release()
            self._cap = None
        self._running = False

        return True

    def frames(self, most_recent=False):
        """Retrieve a new frame from the Webcam and convert it to a
        ColorImage and DepthImage pair.

        Parameters
        ----------
        most_recent: bool
            If true, the OpenCV buffer is emptied for the webcam
            before reading the most recent frame.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`
            The ColorImage and DepthImage of the current frame.
        """
        if most_recent:
            for _ in range(4):
                self._cap.grab()
        for _ in range(1):
            if self._adjust_exposure:
                try:
                    command = [
                        "v4l2-ctl",
                        "-d /dev/video{}".format(self._device_id),
                        "-c exposure_auto=1",
                        "-c exposure_auto_priority=0",
                        "-c exposure_absolute=100",
                        "-c saturation=60" "-c gain=140",
                    ]
                    FNULL = open(os.devnull, "w")
                    subprocess.call(
                        command,
                        stdout=FNULL,
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.SubprocessError:
                    pass
            _, frame = self._cap.read()
        rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return ColorImage(rgb_data, frame=self._frame), None
