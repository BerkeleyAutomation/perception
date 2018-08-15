#!/usr/bin/env python
"""
ROS node that buffers a ROS image stream and allows for grabbing many images simultaneously
"""
import logging
import argparse
import rospy
from rospy import numpy_msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

try:
    from perception.srv import *
except:
    raise RuntimeError("image_buffer unavailable outside of catkin package")

# TODO:
# Timestamps
# Giving a warning if stale data is being returned/delete stale data 
# Launchfile for launching image buffer and primesense camera

# Modify ImageBuffer to work with numpy arrays
ImageBufferResponse = rospy.numpy_msg.numpy_msg(ImageBufferResponse)
ImageBuffer._response_class = ImageBufferResponse

if __name__ == '__main__':
    # Initialize the node.
    rospy.init_node('stream_image_buffer')
    
    # Arguments:
    # instream:       string,             ROS image stream to buffer
    # absolute:       bool, optional      if True, current frame is not prepended to instream (default False)
    # bufsize:        int, optional       Maximum size of image buffer (number of images stored)
    # show_framerate: bool, optional      If True, logs number of frames received in the last 10 seconds
    instream       = rospy.get_param('~instream')
    absolute       = rospy.get_param('~absolute', False)
    bufsize        = rospy.get_param('~bufsize', 100)
    show_framerate = rospy.get_param('~show_framerate', True)
    
    stream_to_buffer = instream
    if not absolute:
        stream_to_buffer = rospy.get_namespace() + stream_to_buffer
    
    # Initialize the CvBridge and image buffer list, as well as misc counting things
    bridge = CvBridge()
    buffer = []
    dtype = 'float32'
    images_so_far = 0
    def callback(data):
        """Callback function for subscribing to an Image topic and creating a buffer
        """
        global dtype
        global images_so_far
            
        # Get cv image (which is a numpy array) from data
        cv_image = bridge.imgmsg_to_cv2(data)
        # Save dtype before we float32-ify it
        dtype = str(cv_image.dtype)
        # Insert and roll buffer
        buffer.insert(0, (np.asarray(cv_image, dtype='float32'), rospy.get_time()))
        if(len(buffer) > bufsize):
            buffer.pop()
        
        # for showing framerate
        images_so_far += 1
                
    # Initialize subscriber with our callback
    rospy.Subscriber(stream_to_buffer, Image, callback)
    
    def handle_request(req):
        """Request-handling for returning a bunch of images stuck together
        """
        # Register time of request
        req_time = rospy.get_time()
        
        # Check if request fits in buffer
        if req.num_requested > len(buffer):
            raise RuntimeError("Number of images requested exceeds current buffer size")
        
        # Cut out the images and timestamps we're returning, save image shape
        ret_images, ret_times = zip(*buffer[:req.num_requested])
        image_shape = ret_images[0].shape
        images_per_frame = 1 if len(image_shape) == 2 else image_shape[2]
        
        # Get timestamps in desired mode
        if req.timing_mode == 0:
            ret_times = np.asarray(ret_times)
        elif req.timing_mode == 1:
            ret_times = np.asarray([req_time - time for time in ret_times])
        else:
            raise RuntimeError("{0} is not a value for timing_mode".format(timing_mode))
        
        # Stack and unravel images because ROS doesn't like multidimensional arrays
        ret_images = np.dstack(ret_images)
        
        return ImageBufferResponse(ret_times, ret_images.ravel(), images_per_frame, dtype, *ret_images.shape)
    
    # Initialize service with our request handler
    s = rospy.Service('stream_image_buffer', ImageBuffer, handle_request)
    
    if show_framerate:
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            rospy.loginfo("{0} frames recorded in the past 10 seconds from {1}".format(images_so_far, stream_to_buffer))
            images_so_far = 0
            r.sleep()
    else:
        rospy.spin()
    
