import numpy as np
import rospy

from autolab_core import Image, ColorImage, DepthImage
from autolab_core.constants import MM_TO_METERS

from .primesense_sensor import PrimesenseSensor

from perception.srv import ImageBuffer, ImageBufferResponse

ImageBufferResponse = rospy.numpy_msg.numpy_msg(ImageBufferResponse)
ImageBuffer._response_class = ImageBufferResponse


class PrimesenseSensor_ROS(PrimesenseSensor):
    """ROS-based version of Primesense RGBD sensor interface

    Requires starting the openni2 ROS driver and the two stream_image_buffer
    (image_buffer.py) ros services for depth and color images. By default,
    the class will look for the depth_image buffer and color_image buffers
    under "{frame}/depth/stream_image_buffer" and
    "{frame}/rgb/stream_image_buffer" respectively (within the current
    ROS namespace).

    This can be changed by passing in depth_image_buffer, color_image_buffer
    (which change where the program looks for the buffer services) and
    depth_absolute, color_absolute (which changes whether the program prepends
    the current ROS namespace).
    """

    def __init__(
        self,
        depth_image_buffer=None,
        depth_absolute=False,
        color_image_buffer=None,
        color_absolute=False,
        flip_images=True,
        frame=None,
        staleness_limit=10.0,
        timeout=10,
    ):
        self._flip_images = flip_images
        self._frame = frame

        self.staleness_limit = staleness_limit
        self.timeout = timeout

        if self._frame is None:
            self._frame = "primesense"
        self._color_frame = "%s_color" % (self._frame)
        self._ir_frame = (
            self._frame
        )  # same as color since we normally use this one

        # Set image buffer locations
        self._depth_image_buffer = (
            "{0}/depth/stream_image_buffer".format(frame)
            if depth_image_buffer is None
            else depth_image_buffer
        )
        self._color_image_buffer = (
            "{0}/rgb/stream_image_buffer".format(frame)
            if color_image_buffer is None
            else color_image_buffer
        )
        if not depth_absolute:
            self._depth_image_buffer = (
                rospy.get_namespace() + self._depth_image_buffer
            )
        if not color_absolute:
            self._color_image_buffer = (
                rospy.get_namespace() + self._color_image_buffer
            )

    def start(self):
        """For PrimesenseSensor, start/stop by launching/stopping
        the associated ROS services"""
        pass

    def stop(self):
        """For PrimesenseSensor, start/stop by launching/stopping
        the associated ROS services"""
        pass

    def _ros_read_images(self, stream_buffer, number, staleness_limit=10.0):
        """Reads images from a stream buffer

        Parameters
        ----------
        stream_buffer : string
            absolute path to the image buffer service
        number : int
            The number of frames to get. Must be less than the image buffer
            service's current buffer size
        staleness_limit : float, optional
            Max value of how many seconds old the oldest image is. If the
            oldest image grabbed is older than this value, a RuntimeError
            is thrown. If None, staleness is ignored.

        Returns
        -------
        List of nump.ndarray objects, each one an image
        Images are in reverse chronological order (newest first)
        """
        rospy.wait_for_service(stream_buffer, timeout=self.timeout)
        ros_image_buffer = rospy.ServiceProxy(stream_buffer, ImageBuffer)
        ret = ros_image_buffer(number, 1)
        if staleness_limit is not None:
            if ret.timestamps[-1] > staleness_limit:
                raise RuntimeError(
                    "Got data {0} seconds old, "
                    "more than allowed {1} seconds".format(
                        ret.timestamps[-1], staleness_limit
                    )
                )

        data = ret.data.reshape(
            ret.data_dim1, ret.data_dim2, ret.data_dim3
        ).astype(ret.dtype)

        # Special handling for 1 element, since dstack's behavior is different
        if number == 1:
            return [data]
        return np.dsplit(data, number)

    @property
    def is_running(self):
        """bool : True if the image buffers are running, or false otherwise.

        Does this by grabbing one frame with staleness checking
        """
        try:
            self.frames()
        except RuntimeError:
            return False
        return True

    def _read_depth_images(self, num_images):
        """Reads depth images from the device"""
        depth_images = self._ros_read_images(
            self._depth_image_buffer, num_images, self.staleness_limit
        )
        for i in range(0, num_images):
            depth_images[i] = (
                depth_images[i] * MM_TO_METERS
            )  # convert to meters
            if self._flip_images:
                depth_images[i] = np.flipud(depth_images[i])
                depth_images[i] = np.fliplr(depth_images[i])
            depth_images[i] = DepthImage(depth_images[i], frame=self._frame)
        return depth_images

    def _read_color_images(self, num_images):
        """Reads color images from the device"""
        color_images = self._ros_read_images(
            self._color_image_buffer, num_images, self.staleness_limit
        )
        for i in range(0, num_images):
            if self._flip_images:
                color_images[i] = np.flipud(color_images[i].astype(np.uint8))
                color_images[i] = np.fliplr(color_images[i].astype(np.uint8))
            color_images[i] = ColorImage(color_images[i], frame=self._frame)
        return color_images

    def _read_depth_image(self):
        """Wrapper to maintain compatibility"""
        return self._read_depth_images(1)[0]

    def _read_color_image(self):
        """Wrapper to maintain compatibility"""
        return self._read_color_images(1)[0]

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
        depths = self._read_depth_images(num_img)

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
        depths = self._read_depth_images(num_img)

        return Image.min_images(depths)
