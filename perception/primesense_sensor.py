"""
Class for interfacing with the Primesense Carmine RGBD sensor
Author: Jeff Mahler
"""
import logging

import numpy as np
from autolab_core import CameraIntrinsics, ColorImage, DepthImage, Image
from autolab_core.constants import MM_TO_METERS
from primesense import openni2

from .camera_sensor import CameraSensor


class PrimesenseRegistrationMode:
    """Primesense registration mode."""

    NONE = 0
    DEPTH_TO_COLOR = 1


class PrimesenseSensor(CameraSensor):
    """Class for interacting with a Primesense RGBD sensor."""

    # Constants for image height and width (in case they're needed somewhere)
    COLOR_IM_HEIGHT = 480
    COLOR_IM_WIDTH = 640
    DEPTH_IM_HEIGHT = 480
    DEPTH_IM_WIDTH = 640
    CENTER_X = float(DEPTH_IM_WIDTH - 1) / 2.0
    CENTER_Y = float(DEPTH_IM_HEIGHT - 1) / 2.0
    FOCAL_X = 525.0
    FOCAL_Y = 525.0
    FPS = 30
    OPENNI2_PATH = "/home/autolab/Libraries/OpenNI-Linux-x64-2.2/Redist"

    def __init__(
        self,
        registration_mode=PrimesenseRegistrationMode.DEPTH_TO_COLOR,
        auto_white_balance=False,
        auto_exposure=True,
        enable_depth_color_sync=True,
        flip_images=True,
        frame=None,
    ):
        self._device = None
        self._depth_stream = None
        self._color_stream = None
        self._running = None

        self._registration_mode = registration_mode
        self._auto_white_balance = auto_white_balance
        self._auto_exposure = auto_exposure
        self._enable_depth_color_sync = enable_depth_color_sync
        self._flip_images = flip_images

        self._frame = frame

        if self._frame is None:
            self._frame = "primesense"
        self._color_frame = "{}_color".format(self._frame)
        self._ir_frame = (
            self._frame
        )  # same as color since we normally use this one

    def __del__(self):
        """Automatically stop the sensor for safety."""
        if self.is_running:
            self.stop()

    @property
    def color_intrinsics(self):
        """:obj:`CameraIntrinsics` : Color camera intrinsics of primesense."""
        return CameraIntrinsics(
            self._ir_frame,
            PrimesenseSensor.FOCAL_X,
            PrimesenseSensor.FOCAL_Y,
            PrimesenseSensor.CENTER_X,
            PrimesenseSensor.CENTER_Y,
            height=PrimesenseSensor.DEPTH_IM_HEIGHT,
            width=PrimesenseSensor.DEPTH_IM_WIDTH,
        )

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : IR camera intrinsics of primesense."""
        return CameraIntrinsics(
            self._ir_frame,
            PrimesenseSensor.FOCAL_X,
            PrimesenseSensor.FOCAL_Y,
            PrimesenseSensor.CENTER_X,
            PrimesenseSensor.CENTER_Y,
            height=PrimesenseSensor.DEPTH_IM_HEIGHT,
            width=PrimesenseSensor.DEPTH_IM_WIDTH,
        )

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise."""
        return self._running

    @property
    def frame(self):
        """:obj:`str` : The reference frame of the sensor."""
        return self._frame

    @property
    def color_frame(self):
        """:obj:`str` : The reference frame of the color sensor."""
        return self._color_frame

    @property
    def ir_frame(self):
        """:obj:`str` : The reference frame of the IR sensor."""
        return self._ir_frame

    def start(self):
        """Start the sensor"""
        # open device
        openni2.initialize(PrimesenseSensor.OPENNI2_PATH)
        self._device = openni2.Device.open_any()

        # open depth stream
        self._depth_stream = self._device.create_depth_stream()
        self._depth_stream.configure_mode(
            PrimesenseSensor.DEPTH_IM_WIDTH,
            PrimesenseSensor.DEPTH_IM_HEIGHT,
            PrimesenseSensor.FPS,
            openni2.PIXEL_FORMAT_DEPTH_1_MM,
        )
        self._depth_stream.start()

        # open color stream
        self._color_stream = self._device.create_color_stream()
        self._color_stream.configure_mode(
            PrimesenseSensor.COLOR_IM_WIDTH,
            PrimesenseSensor.COLOR_IM_HEIGHT,
            PrimesenseSensor.FPS,
            openni2.PIXEL_FORMAT_RGB888,
        )
        self._color_stream.camera.set_auto_white_balance(
            self._auto_white_balance
        )
        self._color_stream.camera.set_auto_exposure(self._auto_exposure)
        self._color_stream.start()

        # configure device
        if (
            self._registration_mode
            == PrimesenseRegistrationMode.DEPTH_TO_COLOR
        ):
            self._device.set_image_registration_mode(
                openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
            )
        else:
            self._device.set_image_registration_mode(
                openni2.IMAGE_REGISTRATION_OFF
            )

        self._device.set_depth_color_sync_enabled(
            self._enable_depth_color_sync
        )

        self._running = True

    def stop(self):
        """Stop the sensor"""
        # check that everything is running
        if not self._running or self._device is None:
            logging.warning("Primesense not running. Aborting stop")
            return False

        # stop streams
        if self._depth_stream:
            self._depth_stream.stop()
        if self._color_stream:
            self._color_stream.stop()
        self._running = False

        # Unload openni2
        openni2.unload()
        return True

    def _read_depth_image(self):
        """Reads a depth image from the device"""
        # read raw uint16 buffer
        im_arr = self._depth_stream.read_frame()
        raw_buf = im_arr.get_buffer_as_uint16()
        buf_array = np.array(
            [
                raw_buf[i]
                for i in range(
                    PrimesenseSensor.DEPTH_IM_WIDTH
                    * PrimesenseSensor.DEPTH_IM_HEIGHT
                )
            ]
        )

        # convert to image in meters
        depth_image = buf_array.reshape(
            PrimesenseSensor.DEPTH_IM_HEIGHT, PrimesenseSensor.DEPTH_IM_WIDTH
        )
        depth_image = depth_image * MM_TO_METERS  # convert to meters
        if self._flip_images:
            depth_image = np.flipud(depth_image)
        else:
            depth_image = np.fliplr(depth_image)
        return DepthImage(depth_image, frame=self._frame)

    def _read_color_image(self):
        """Reads a color image from the device"""
        # read raw buffer
        im_arr = self._color_stream.read_frame()
        raw_buf = im_arr.get_buffer_as_triplet()
        r_array = np.array(
            [
                raw_buf[i][0]
                for i in range(
                    PrimesenseSensor.COLOR_IM_WIDTH
                    * PrimesenseSensor.COLOR_IM_HEIGHT
                )
            ]
        )
        g_array = np.array(
            [
                raw_buf[i][1]
                for i in range(
                    PrimesenseSensor.COLOR_IM_WIDTH
                    * PrimesenseSensor.COLOR_IM_HEIGHT
                )
            ]
        )
        b_array = np.array(
            [
                raw_buf[i][2]
                for i in range(
                    PrimesenseSensor.COLOR_IM_WIDTH
                    * PrimesenseSensor.COLOR_IM_HEIGHT
                )
            ]
        )

        # convert to uint8 image
        color_image = np.zeros(
            [
                PrimesenseSensor.COLOR_IM_HEIGHT,
                PrimesenseSensor.COLOR_IM_WIDTH,
                3,
            ]
        )
        color_image[:, :, 0] = r_array.reshape(
            PrimesenseSensor.COLOR_IM_HEIGHT, PrimesenseSensor.COLOR_IM_WIDTH
        )
        color_image[:, :, 1] = g_array.reshape(
            PrimesenseSensor.COLOR_IM_HEIGHT, PrimesenseSensor.COLOR_IM_WIDTH
        )
        color_image[:, :, 2] = b_array.reshape(
            PrimesenseSensor.COLOR_IM_HEIGHT, PrimesenseSensor.COLOR_IM_WIDTH
        )
        if self._flip_images:
            color_image = np.flipud(color_image.astype(np.uint8))
        else:
            color_image = np.fliplr(color_image.astype(np.uint8))
        return ColorImage(color_image, frame=self._frame)

    def frames(self):
        """Retrieve a new frame from the Kinect and convert it to a
        ColorImage and a DepthImage.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`
            The ColorImage and DepthImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        color_im = self._read_color_image()
        depth_im = self._read_depth_image()
        return color_im, depth_im

    def median_depth_img(self, num_img=1, fill_depth=0.0):
        """Collect a series of depth images and return the median of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The median DepthImage collected from the frames.
        """
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        median_depth = Image.median_images(depths)
        median_depth.data[median_depth.data == 0.0] = fill_depth
        return median_depth

    def min_depth_img(self, num_img=1):
        """Collect a series of depth images and return the min of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The min DepthImage collected from the frames.
        """
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        return Image.min_images(depths)
