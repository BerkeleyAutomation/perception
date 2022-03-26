"""
Class for interfacing with the Primesense RGBD sensor
Author: Jeff Mahler
"""
import logging
import time

import numpy as np
import rospy
import sensor_msgs.msg
from autolab_core import CameraIntrinsics, ColorImage, DepthImage, Image
from autolab_core.constants import MM_TO_METERS
from cv_bridge import CvBridge, CvBridgeError

from .camera_sensor import CameraSensor


class Kinect2BridgedQuality:
    """Kinect quality for bridged mode"""

    HD = "hd"
    QUARTER_HD = "qhd"
    SD = "sd"


class KinectSensorBridged(CameraSensor):
    """Class for interacting with a Kinect v2 RGBD sensor through the kinect
    bridge https://github.com/code-iai/iai_kinect2. This is preferrable for
    visualization and debug because the kinect bridge will continuously
    publish image and point cloud info.
    """

    def __init__(
        self,
        quality=Kinect2BridgedQuality.HD,
        frame="kinect2_rgb_optical_frame",
    ):
        """Initialize a Kinect v2 sensor which connects to the
        iai_kinect2 bridge
        ----------
        quality : :obj:`str`
            The quality (HD, Quarter-HD, SD) of the image data that
            should be subscribed to
        frame : :obj:`str`
            The name of the frame of reference in which the sensor resides.
            If None, this will be set to 'kinect2_rgb_optical_frame'
        """
        # set member vars
        self._frame = frame

        self.topic_image_color = "/kinect2/%s/image_color_rect" % (quality)
        self.topic_image_depth = "/kinect2/%s/image_depth_rect" % (quality)
        self.topic_info_camera = "/kinect2/%s/camera_info" % (quality)

        self._initialized = False
        self._format = None
        self._camera_intr = None
        self._cur_depth_im = None
        self._running = False
        self._bridge = CvBridge()

    def __del__(self):
        """Automatically stop the sensor for safety."""
        if self.is_running:
            self.stop()

    def _set_camera_properties(self, msg):
        """Set the camera intrinsics from an info msg."""
        focal_x = msg.K[0]
        focal_y = msg.K[4]
        center_x = msg.K[2]
        center_y = msg.K[5]
        im_height = msg.height
        im_width = msg.width
        self._camera_intr = CameraIntrinsics(
            self._frame,
            focal_x,
            focal_y,
            center_x,
            center_y,
            height=im_height,
            width=im_width,
        )

    def _process_image_msg(self, msg):
        """Process an image message and return a numpy array with the image data
        Returns
        -------
        :obj:`numpy.ndarray` containing the image in the image message

        Raises
        ------
        CvBridgeError
            If the bridge is not able to convert the image
        """
        encoding = msg.encoding
        try:
            image = self._bridge.imgmsg_to_cv2(msg, encoding)
        except CvBridgeError as e:
            rospy.logerr(e)
        return image

    def _color_image_callback(self, image_msg):
        """subscribe to image topic and keep it up to date"""
        color_arr = self._process_image_msg(image_msg)
        self._cur_color_im = ColorImage(color_arr[:, :, ::-1], self._frame)

    def _depth_image_callback(self, image_msg):
        """subscribe to depth image topic and keep it up to date"""
        encoding = image_msg.encoding
        try:
            depth_arr = self._bridge.imgmsg_to_cv2(image_msg, encoding)
        except CvBridgeError as e:
            rospy.logerr(e)
        depth = np.array(depth_arr * MM_TO_METERS, np.float32)
        self._cur_depth_im = DepthImage(depth, self._frame)

    def _camera_info_callback(self, msg):
        """Callback for reading camera info."""
        self._camera_info_sub.unregister()
        self._set_camera_properties(msg)

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : IR camera intrinsics of Kinect."""
        return self._camera_intr

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise."""
        return self._running

    @property
    def frame(self):
        """:obj:`str` : The reference frame of the sensor."""
        return self._frame

    def start(self):
        """Start the sensor"""
        # initialize subscribers
        self._image_sub = rospy.Subscriber(
            self.topic_image_color,
            sensor_msgs.msg.Image,
            self._color_image_callback,
        )
        self._depth_sub = rospy.Subscriber(
            self.topic_image_depth,
            sensor_msgs.msg.Image,
            self._depth_image_callback,
        )
        self._camera_info_sub = rospy.Subscriber(
            self.topic_info_camera,
            sensor_msgs.msg.CameraInfo,
            self._camera_info_callback,
        )

        timeout = 10
        try:
            rospy.loginfo("waiting to recieve a message from the Kinect")
            rospy.wait_for_message(
                self.topic_image_color, sensor_msgs.msg.Image, timeout=timeout
            )
            rospy.wait_for_message(
                self.topic_image_depth, sensor_msgs.msg.Image, timeout=timeout
            )
            rospy.wait_for_message(
                self.topic_info_camera,
                sensor_msgs.msg.CameraInfo,
                timeout=timeout,
            )
        except rospy.ROSException as e:
            print("KINECT NOT FOUND")
            rospy.logerr("Kinect topic not found, Kinect not started")
            rospy.logerr(e)

        while self._camera_intr is None:
            time.sleep(0.1)

        self._running = True

    def stop(self):
        """Stop the sensor"""
        # check that everything is running
        if not self._running:
            logging.warning("Kinect not running. Aborting stop")
            return False

        # stop subs
        self._image_sub.unregister()
        self._depth_sub.unregister()
        self._camera_info_sub.unregister

        self._running = False
        return True

    def frames(self):
        """Retrieve a new frame from the Kinect and convert it to a ColorImage,
        a DepthImage is always none for this type

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`
            The ColorImage and DepthImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        # wait for a new image
        while self._cur_depth_im is None or self._cur_color_im is None:
            time.sleep(0.01)

        # read next image
        depth_im = self._cur_depth_im
        color_im = self._cur_color_im

        self._cur_color_im = None
        self._cur_depth_im = None

        # TODO add ir image
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
