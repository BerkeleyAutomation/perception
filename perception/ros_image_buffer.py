#!/usr/bin/env python
import rospy
from rospy import numpy_msg
import argparse
from sensor_msgs.msg import Image
from perception.srv import *
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

ImageBufferResponse = rospy.numpy_msg.numpy_msg(ImageBufferResponse)
ImageBuffer._response_class = ImageBufferResponse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a ROS image buffer")
    parser.add_argument('instream', metavar='in', type=str, help="The stream to buffer")
    parser.add_argument('-a', '--absolute', dest='absolute', action='store_true',
                        help="Make the path to the ROS stream absolute (do not prepend ROS_NAMESPACE)")
    parser.add_argument('--bufsize', dest='bufsize', default=100, type=int,
                        help="Specify the number of images to buffer (default 100)")
    parser.add_argument('-r', '--show_framerate', dest='show_framerate', action='store_true',
                        help="Show a framerate message every 10 seconds")
    args = parser.parse_args()
    
    bufsize = args.bufsize
    stream_to_buffer = args.instream
    if not args.absolute:
        stream_to_buffer = rospy.get_namespace() + stream_to_buffer
    
    rospy.init_node('stream_image_buffer')
    
    bridge = CvBridge()
    buffer_of_images = []
    if args.show_framerate:
        dtype = 'float32'
        images_so_far = 0
        time_old = rospy.get_rostime()
        ten_secs = rospy.Duration(10, 0)
    def callback(data):
        global dtype
        if args.show_framerate:
            global images_so_far
            global time_old
            global ten_secs
        try:
            cv_image = bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
        dtype = str(cv_image.dtype)
        buffer_of_images.insert(0, np.asarray(cv_image, dtype='float32'))
        if(len(buffer_of_images) > bufsize):
            buffer_of_images.pop()
        
        if args.show_framerate:
            images_so_far += 1
            time_new = rospy.get_rostime()
            if time_new - time_old >= ten_secs:
                print("{0} frames recorded in the past 10 seconds".format(images_so_far))
                time_old = time_new
                images_so_far = 0

    rospy.Subscriber(stream_to_buffer, Image, callback)
    
    def handle_request(req):
        to_return = buffer_of_images[:min(bufsize, req.num_requested)]
        image_shape = to_return[0].shape
        images_per_frame = 1 if len(image_shape) == 2 else image_shape[2]
        
        ret = np.dstack(to_return)
        return ImageBufferResponse(images_per_frame, dtype, ret.ravel(), *ret.shape)
    
    s = rospy.Service('stream_image_buffer', ImageBuffer, handle_request)
    
    rospy.spin()
    