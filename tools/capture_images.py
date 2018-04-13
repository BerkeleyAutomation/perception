"""
Script to capture a set of test images.
Be sure to register beforehand!!!
Author: Jeff Mahler
"""
import argparse
import copy
import IPython
import numpy as np
import os
import logging
import rospy
import sys

import matplotlib.pyplot as plt

from autolab_core import RigidTransform, Box, YamlConfig
from perception import RgbdSensorFactory, Image

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)
    rospy.init_node('capture_images', anonymous=True)

    # parse args
    parser = argparse.ArgumentParser(description='Capture a set of RGB-D images from a set of sensors')
    parser.add_argument('name', type=str, help='name for test files')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/capture_images.yaml', help='path to configuration file to use')
    args = parser.parse_args()
    name = args.name
    config_filename = args.config_filename

    # read config
    config = YamlConfig(config_filename)
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    for sensor_name, sensor_config in config['sensors'].iteritems():
        print('Capturing images from sensor %s' %(sensor_name))
        save_dir = os.path.join(output_dir, sensor_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)        
        
        # read params
        sensor_type = sensor_config['type']
        sensor_frame = sensor_config['frame']
        
        # read camera calib
        tf_filename = '%s_to_world.tf' %(sensor_frame)
        T_camera_world = RigidTransform.load(os.path.join(config['calib_dir'], sensor_frame, tf_filename))
        T_camera_world.save(os.path.join(save_dir, tf_filename))

        # setup sensor
        sensor = RgbdSensorFactory.sensor(sensor_type, sensor_config)

        # start the sensor
        sensor.start()
        ir_intrinsics = sensor.ir_intrinsics
        ir_intrinsics.save(os.path.join(save_dir, '%s.intr' %(sensor.frame)))

        # get raw images
        for i in range(sensor_config['num_images']):
            logging.info('Capturing image %d' %(i))
            color, depth, ir = sensor.frames()
            color.save(os.path.join(save_dir, 'color_%d.png' %(i)))
            depth.save(os.path.join(save_dir, 'depth_%d.npy' %(i)))
            if ir is not None:
                ir.save(os.path.join(save_dir, 'ir_%d.npy' %(i)))

        sensor.stop()
