#!/usr/bin/env python
"""
Interface to the Ensenso N* Sensor
Author: Jeff Mahler
"""
import IPython
import logging
import numpy as np
import os
import struct
import sys
import time
import signal

try:
    from cv_bridge import CvBridge, CvBridgeError
    import rospy
    import sensor_msgs.msg
    import sensor_msgs.point_cloud2 as pc2
except ImportError:
    logging.warning("Failed to import ROS in Kinect2_sensor.py. Kinect will not be able to be used in bridged mode")
    
from .constants import MM_TO_METERS, INTR_EXTENSION
from . import CameraIntrinsics, CameraSensor, ColorImage, DepthImage, Image



class KinectSensorBridged(CameraSensor):
    """ Class for interfacing with an Ensenso N* sensor.
    """
    QUALITY = Kinect2BridgedQuality.HD
    TOPIC_IMAGE_COLOR = '/kinect2/%s/image_color' %(QUALITY)
    TOPIC_IMAGE_DEPTH = '/kinect2/%s/image_depth_rect' %(QUALITY)
    TOPIC_INFO_CAMERA = '/kinect2/%s/camera_info' %(QUALITY)

    def __init__(self, quality=Kinect2Quality.HD, frame='ensenso'):
        # set member vars
        self._frame = frame
        self._quality = quality
        
        self._initialized = False
        self._format = None
        self._camera_intr = None
        self._cur_depth_im = None
        self._running = False
        self._bridge = CvBridge()
        
    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()
            
    def _set_camera_properties(self, msg):
        """ Set the camera intrinsics from an info msg. """
        focal_x = msg.K[0]
        focal_y = msg.K[4]
        center_x = msg.K[2]
        center_y = msg.K[5]
        im_height = msg.height
        im_width = msg.width
        self._camera_intr = CameraIntrinsics(self._frame, focal_x, focal_y,
                                             center_x, center_y,
                                             height=im_height,
                                             width=im_width)

    def _process_image_msg(self, msg):
        """ Process an image message and return a numpy array with the image data
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
        """ subscribe to image topic and keep it up to date
        """
        color_arr = self._process_image_msg(image_msg)
        self._cur_color_im = ColorImage(color_arr[:,:,::-1], self._frame)
 
    def _depth_image_callback(self, image_msg):
        """ subscribe to depth image topic and keep it up to date
        """
        depth_arr = self._process_image_msg(image_msg)
        depth = np.array(depth_arr, np.float32)
        self._cur_depth_im = DepthImage(depth, self._frame)
        
    def _camera_info_callback(self, msg):
        """ Callback for reading camera info. """
        self._camera_info_sub.unregister()
        self._set_camera_properties(msg)
    
    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Ensenso IR camera.
        """
        return self._camera_intr

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise.
        """
        return self._running

    @property
    def frame(self):
        """:obj:`str` : The reference frame of the sensor.
        """
        return self._frame

    def start(self):
        """ Start the sensor """
        # initialize subscribers
        self._image_sub = rospy.Subscriber(KinectSensorBridged.TOPIC_IMAGE_COLOR, sensor_msgs.msg.Image, self._color_image_callback)
        self._depth_sub = rospy.Subscriber(KinectSensorBridged.TOPIC_IMAGE_DEPTH, sensor_msgs.msg.Image, self._depth_image_callback)
        self._camera_info_sub = rospy.Subscriber(KinectSensorBridged.TOPIC_INFO_CAMERA, sensor_msgs.msg.CameraInfo, self._camera_info_callback)
        
        timeout = 10
        try:
            logging.info("waiting to recieve a message from the Kinect")
            rospy.wait_for_message(KinectSensorBridged.TOPIC_IMAGE_COLOR, sensor_msgs.msg.Image, timeout=timeout)
            rospy.wait_for_message(KinectSensorBridged.TOPIC_IMAGE_DEPTH, sensor_msgs.msg.Image, timeout=timeout)
            rospy.wait_for_message(KinectSensorBridged.TOPIC_INFO_CAMERA, sensor_msgs.msg.CameraInfo, timeout=timeout)
        except rospy.ROSException as e:
            logging.error("Kinect topic not found, Kinect not started")
            logging.error(e)

        while self._camera_intr is None:
            time.sleep(0.1)
        
        self._running = True

    def stop(self):
        """ Stop the sensor """
        # check that everything is running
        if not self._running:
            logging.warning('Kinect not running. Aborting stop')
            return False

        # stop subs
        self._image_sub.unregister()
        self._depth_sub.unregister()
        self._camera_info_sub.unregister

        self._running = False
        return True

    def frames(self):
        """Retrieve a new frame from the Ensenso and convert it to a ColorImage,
        a DepthImage, IrImage is always none for this type

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

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

        #TODO add ir image
        return color_im, depth_im, None

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
        
def main(args):
    # from visualization import Visualizer2D as vis2d
    # from visualization import Visualizer3D as vis3d
    import matplotlib.pyplot as vis2d

    # set logging
    logging.getLogger().setLevel(logging.DEBUG)
    rospy.init_node('kinect_reader', anonymous=True)

    num_frames = 5
    sensor = KinectSensorBridge()
    sensor.start()
    def handler(signum, frame):
        rospy.loginfo('caught CTRL+C, exiting...')        
        if sensor is not None:
            sensor.stop()            
        exit(0)
    signal.signal(signal.SIGINT, handler)

    total_time = 0
    for i in range(num_frames):        
        if i > 0:
            start_time = time.time()

        _, depth_im, _ = sensor.frames()

        if i > 0:
            total_time += time.time() - start_time
            logging.info('Frame %d' %(i))
            logging.info('Avg FPS: %.5f' %(float(i) / total_time))
        
    depth_im = sensor.median_depth_img(num_img=5)
    color_im, depth_im, _ = sensor.frames()

    sensor.stop()

    vis2d.figure()
    vis2d.subplot('211')
    vis2d.imshow(depth_im.data)
    vis2d.title('Kinect - depth Raw')
    
    vis2d.subplot('212')
    vis2d.imshow(color_im.data)
    vis2d.title("kinect color")
    vis2d.show()
    
if __name__ == '__main__':
    main(sys.argv)
