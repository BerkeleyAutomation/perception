"""
Class for interfacing with the Primesense RGBD sensor
Author: Jeff Mahler
"""
import logging

import numpy as np
import pylibfreenect2 as lf2
from autolab_core import (
    CameraIntrinsics,
    ColorImage,
    DepthImage,
    Image,
    IrImage,
)
from autolab_core.constants import MM_TO_METERS

from .camera_sensor import CameraSensor


class Kinect2PacketPipelineMode:
    """Type of pipeline for Kinect packet processing."""

    OPENGL = 0
    CPU = 1
    OPENCL = 2
    AUTO = 3


class Kinect2FrameMode:
    """Type of frames that Kinect processes."""

    COLOR_DEPTH = 0
    COLOR_DEPTH_IR = 1


class Kinect2RegistrationMode:
    """Kinect registration mode."""

    NONE = 0
    COLOR_TO_DEPTH = 1


class Kinect2DepthMode:
    """Kinect depth mode setting."""

    METERS = 0
    MILLIMETERS = 1


class Kinect2Sensor(CameraSensor):
    # constants for image height and width (in case they're needed somewhere)
    """Class for interacting with a Kinect v2 RGBD sensor directly through
    protonect driver. https://github.com/OpenKinect/libfreenect2
    """

    # Constants for image height and width (in case they're needed somewhere)
    COLOR_IM_HEIGHT = 1080
    COLOR_IM_WIDTH = 1920
    DEPTH_IM_HEIGHT = 424
    DEPTH_IM_WIDTH = 512

    def __init__(
        self,
        packet_pipeline_mode=Kinect2PacketPipelineMode.AUTO,
        registration_mode=Kinect2RegistrationMode.COLOR_TO_DEPTH,
        depth_mode=Kinect2DepthMode.METERS,
        device_num=0,
        frame=None,
    ):
        """Initialize a Kinect v2 sensor directly to the protonect driver with
        the given configuration. When kinect is connected to the protonect
        driver directly, the iai_kinect kinect_bridge cannot be run at the
        same time.

        Parameters
        ----------
        packet_pipeline_mode : int
            Either Kinect2PacketPipelineMode.OPENGL,
            Kinect2PacketPipelineMode.OPENCL or
            Kinect2PacketPipelineMode.CPU -- indicates packet processing type.
            If not specified the packet pipeline will be determined
            automatically.

        registration_mode : int
            Either Kinect2RegistrationMode.NONE or
            Kinect2RegistrationMode.COLOR_TO_DEPT -- The mode for registering
            a color image to the IR camera frame of reference.

        depth_mode : int
            Either Kinect2DepthMode.METERS or Kinect2DepthMode.MILLIMETERS --
            the units for depths returned from the Kinect frame arrays.

        device_num : int
            The sensor's device number on the USB bus.

        frame : :obj:`str`
            The name of the frame of reference in which the sensor resides.
            If None, this will be set to 'kinect2_num', where num is replaced
            with the device number.
        """
        self._device = None
        self._running = False
        self._packet_pipeline_mode = packet_pipeline_mode
        self._registration_mode = registration_mode
        self._depth_mode = depth_mode
        self._device_num = device_num
        self._frame = frame

        if self._frame is None:
            self._frame = "kinect2_%d" % (self._device_num)
        self._color_frame = "%s_color" % (self._frame)
        self._ir_frame = (
            self._frame
        )  # same as color since we normally use this one

    def __del__(self):
        """Automatically stop the sensor for safety."""
        if self.is_running:
            self.stop()

    @property
    def color_intrinsics(self):
        """:obj:`CameraIntrinsics` : Color camera intrinsics of Kinect."""
        if self._device is None:
            raise RuntimeError(
                "Kinect2 device not runnning. Cannot return color intrinsics"
            )
        camera_params = self._device.getColorCameraParams()
        return CameraIntrinsics(
            self._color_frame,
            camera_params.fx,
            camera_params.fy,
            camera_params.cx,
            camera_params.cy,
        )

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : IR camera intrinsics for the Kinect."""
        if self._device is None:
            raise RuntimeError(
                "Kinect2 device not runnning. Cannot return IR intrinsics"
            )
        camera_params = self._device.getIrCameraParams()
        return CameraIntrinsics(
            self._ir_frame,
            camera_params.fx,
            camera_params.fy,
            camera_params.cx,
            camera_params.cy,
            height=Kinect2Sensor.DEPTH_IM_HEIGHT,
            width=Kinect2Sensor.DEPTH_IM_WIDTH,
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
        """Starts the Kinect v2 sensor stream.

        Raises
        ------
        IOError
            If the Kinect v2 is not detected.
        """

        # setup logger
        self._logger = lf2.createConsoleLogger(lf2.LoggerLevel.Warning)
        lf2.setGlobalLogger(self._logger)

        # open packet pipeline
        self._pipeline = None
        if (
            self._packet_pipeline_mode == Kinect2PacketPipelineMode.OPENGL
            or self._packet_pipeline_mode == Kinect2PacketPipelineMode.AUTO
        ):
            # Try OpenGL packet pipeline first or if specified
            try:
                self._pipeline = lf2.OpenGLPacketPipeline()
            except BaseException:
                logging.warning(
                    "OpenGL not available. "
                    "Defaulting to CPU-based packet pipeline."
                )

        if self._pipeline is None and (
            self._packet_pipeline_mode == Kinect2PacketPipelineMode.OPENCL
            or self._packet_pipeline_mode == Kinect2PacketPipelineMode.AUTO
        ):
            # Try OpenCL if available
            try:
                self._pipeline = lf2.OpenCLPacketPipeline()
            except BaseException:
                logging.warning(
                    "OpenCL not available. Defaulting to CPU packet pipeline."
                )

        if (
            self._pipeline is None
            or self._packet_pipeline_mode == Kinect2PacketPipelineMode.CPU
        ):  # CPU packet pipeline
            self._pipeline = lf2.CpuPacketPipeline()

        # check devices
        self._fn_handle = lf2.Freenect2()
        self._num_devices = self._fn_handle.enumerateDevices()
        if self._num_devices == 0:
            raise IOError(
                "Failed to start stream. No Kinect2 devices available!"
            )
        if self._num_devices <= self._device_num:
            raise IOError(
                "Failed to start stream. Device num %d unavailable!"
                % (self._device_num)
            )

        # open device
        self._serial = self._fn_handle.getDeviceSerialNumber(self._device_num)
        self._device = self._fn_handle.openDevice(
            self._serial, pipeline=self._pipeline
        )

        # add device sync modes
        self._listener = lf2.SyncMultiFrameListener(
            lf2.FrameType.Color | lf2.FrameType.Ir | lf2.FrameType.Depth
        )
        self._device.setColorFrameListener(self._listener)
        self._device.setIrAndDepthFrameListener(self._listener)

        # start device
        self._device.start()

        # open registration
        self._registration = None
        if self._registration_mode == Kinect2RegistrationMode.COLOR_TO_DEPTH:
            logging.debug("Using color to depth registration")
            self._registration = lf2.Registration(
                self._device.getIrCameraParams(),
                self._device.getColorCameraParams(),
            )
        self._running = True

    def stop(self):
        """Stops the Kinect2 sensor stream.

        Returns
        -------
        bool
            True if the stream was stopped, False if the device was already
            stopped or was not otherwise available.
        """
        # check that everything is running
        if not self._running or self._device is None:
            logging.warning(
                "Kinect2 device %d not runnning. Aborting stop"
                % (self._device_num)
            )
            return False

        # stop the device
        self._device.stop()
        self._device.close()
        self._device = None
        self._running = False
        return True

    def frames(self, skip_registration=False):
        """Retrieve a new frame from the Kinect and convert it to a
        ColorImage and a DepthImage

        Parameters
        ----------
        skip_registration : bool
            If True, the registration step is skipped.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`
            The ColorImage and DepthImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        color_im, depth_im, _, _ = self._frames_and_index_map(
            skip_registration=skip_registration
        )
        return color_im, depth_im

    def median_depth_img(self, num_img=1):
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

        return Image.median_images(depths)

    def _frames_and_index_map(self, skip_registration=False):
        """Retrieve a new frame from the Kinect and return a ColorImage,
        DepthImage, IrImage, and a map from depth pixels to color
        pixel indices.

        Parameters
        ----------
        skip_registration : bool
            If True, the registration step is skipped.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`,
          :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the
            current frame, and an ndarray that maps pixels
            of the depth image to the index of the
            corresponding pixel in the color image.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        if not self._running:
            raise RuntimeError(
                "Kinect2 device %s not runnning. Cannot read frames"
                % (self._device_num)
            )

        # read frames
        frames = self._listener.waitForNewFrame()
        unregistered_color = frames["color"]
        distorted_depth = frames["depth"]
        ir = frames["ir"]

        # apply color to depth registration
        color_frame = self._color_frame
        color = unregistered_color
        depth = distorted_depth
        color_depth_map = (
            np.zeros([depth.height, depth.width]).astype(np.int32).ravel()
        )
        if (
            not skip_registration
            and self._registration_mode
            == Kinect2RegistrationMode.COLOR_TO_DEPTH
        ):
            color_frame = self._ir_frame
            depth = lf2.Frame(
                depth.width, depth.height, 4, lf2.FrameType.Depth
            )
            color = lf2.Frame(
                depth.width, depth.height, 4, lf2.FrameType.Color
            )
            self._registration.apply(
                unregistered_color,
                distorted_depth,
                depth,
                color,
                color_depth_map=color_depth_map,
            )

        # convert to array (copy needed to prevent reference of deleted data
        color_arr = np.copy(color.asarray())
        color_arr[:, :, [0, 2]] = color_arr[:, :, [2, 0]]  # convert BGR to RGB
        color_arr[:, :, 0] = np.fliplr(color_arr[:, :, 0])
        color_arr[:, :, 1] = np.fliplr(color_arr[:, :, 1])
        color_arr[:, :, 2] = np.fliplr(color_arr[:, :, 2])
        color_arr[:, :, 3] = np.fliplr(color_arr[:, :, 3])
        depth_arr = np.fliplr(np.copy(depth.asarray()))
        ir_arr = np.fliplr(np.copy(ir.asarray()))

        # convert from mm to meters
        if self._depth_mode == Kinect2DepthMode.METERS:
            depth_arr = depth_arr * MM_TO_METERS

        # Release and return
        self._listener.release(frames)
        return (
            ColorImage(color_arr[:, :, :3], color_frame),
            DepthImage(depth_arr, self._ir_frame),
            IrImage(ir_arr.astype(np.uint16), self._ir_frame),
            color_depth_map,
        )
