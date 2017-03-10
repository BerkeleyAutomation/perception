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
    args = parser.parse_args()
    
    bufsize = args.bufsize
    stream_to_buffer = args.instream
    if not args.absolute:
        stream_to_buffer = rospy.get_namespace() + stream_to_buffer
    
    rospy.init_node('{0}/stream_image_buffer'.format(stream_to_buffer))
    
    buffer_of_images = []
    def callback(data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
        buffer_of_images.insert(0, np.asArray(cv_image, dtype='float32'))
        if(len(buffer_of_images) > bufsize):
            buffer_of_images.pop()

    rospy.Subscriber(stream_to_buffer, Image, callback)
    
    def handle_request(req):
        to_return = buffer_of_images[:min(bufsize, req.num_requested)]
        image_shape = to_return[0].shape
        images_per_frame = 1 if len(images_shape) == 2 else image_shape[2]
        return ImageBufferResponse(images_per_frame, np.dstack((to_return)))
    
    rospy.spin()
    