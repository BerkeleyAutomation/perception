#!/usr/bin/env python
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
    # Enable logger
    logging.getLogger().setLevel(logging.INFO)
    
    # Initialize and run argument parser 
    parser = argparse.ArgumentParser(description="Initialize a ROS image buffer")
    parser.add_argument('instream', metavar='in', type=str, help="The stream to buffer")
    parser.add_argument('-a', '--absolute', dest='absolute', action='store_true',
                        help="Make the path to the ROS stream absolute (do not prepend ROS_NAMESPACE)")
    parser.add_argument('--bufsize', dest='bufsize', default=100, type=int,
                        help="Specify the number of images to buffer (default 100)")
    parser.add_argument('-r', '--show_framerate', dest='show_framerate', action='store_true',
                        help="Show a framerate message every 10 seconds")
    args = parser.parse_args()
    
    # Set buffer size and target stream
    bufsize = args.bufsize
    stream_to_buffer = args.instream
    if not args.absolute:
        stream_to_buffer = rospy.get_namespace() + stream_to_buffer
    
    # Initialize the node. This is overwritten most of the time by roslaunch
    rospy.init_node('stream_image_buffer')
    
    # Initialize the CvBridge and image buffer list, as well as misc counting things
    bridge = CvBridge()
    buffer_of_images = []
    dtype = 'float32'
    if args.show_framerate:
        images_so_far = 0
        time_old = rospy.get_rostime()
        ten_secs = rospy.Duration(10, 0)
    def callback(data):
        """Callback function for subscribing to an Image topic and creating a buffer
        """
        global dtype
        if args.show_framerate:
            global images_so_far
            global time_old
            global ten_secs
            
        # Get cv image (which is a numpy array) from data
        cv_image = bridge.imgmsg_to_cv2(data)
        # Save dtype before we float32-ify it
        dtype = str(cv_image.dtype)
        # Insert and roll buffer
        buffer_of_images.insert(0, np.asarray(cv_image, dtype='float32'))
        if(len(buffer_of_images) > bufsize):
            buffer_of_images.pop()
        
        # Stuff for showing framerate if that option is checked
        if args.show_framerate:
            images_so_far += 1
            time_new = rospy.get_rostime()
            if time_new - time_old >= ten_secs:
                print("{0} frames recorded in the past 10 seconds".format(images_so_far))
                time_old = time_new
                images_so_far = 0
                
    # Initialize subscriber with our callback
    rospy.Subscriber(stream_to_buffer, Image, callback)
    
    def handle_request(req):
        """Request-handling for returning a bunch of images stuck together
        """
        # Check if request fits in buffer
        if req.num_requested > len(buffer_of_images):
            raise RuntimeError("Number of images requested exceeds buffer size")
        
        # Cut out the images we're returning, save their shape
        to_return = buffer_of_images[:min(bufsize, req.num_requested)]
        image_shape = to_return[0].shape
        images_per_frame = 1 if len(image_shape) == 2 else image_shape[2]
        
        # Stack and unravel images because ROS doesn't like multidimensional arrays
        ret = np.dstack(to_return)
        return ImageBufferResponse(images_per_frame, dtype, ret.ravel(), *ret.shape)
    
    # Initialize service with our request handler
    s = rospy.Service('stream_image_buffer', ImageBuffer, handle_request)
    
    rospy.spin()
    