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
    
from perception import CameraIntrinsics, CameraSensor, ColorImage, DepthImage, Image, RgbdSensorFactory, Kinect2BridgedQuality
       
def main(args):
    # from visualization import Visualizer2D as vis2d
    # from visualization import Visualizer3D as vis3d
    import matplotlib.pyplot as vis2d

    # set logging
    logging.getLogger().setLevel(logging.DEBUG)
    rospy.init_node('kinect_reader', anonymous=True)

    num_frames = 5
    sensor_cfg = {"quality": Kinect2BridgedQuality.HD, "frame":'kinect2_rgb_optical_frame'}
    sensor_type = "bridged_kinect2"
    sensor = RgbdSensorFactory.sensor(sensor_type, sensor_cfg)
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
