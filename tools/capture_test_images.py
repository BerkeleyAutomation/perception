"""
Script to capture a set of test images.
Be sure to register beforehand!!!
Author: Jeff Mahler
"""
import copy
import IPython
import numpy as np
import os
import logging
import sys
import argparse

import matplotlib.pyplot as plt

from autolab_core import RigidTransform, Box, YamlConfig
from perception import RgbdSensorFactory, Image

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Capture a set of test images from the Kinect2')
    parser.add_argument('output_dir', type=str, help='location to save the images')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/capture_test_images.yaml', help='path to configuration file to use')
    args = parser.parse_args()
    output_dir = args.output_dir
    config_filename = args.config_filename

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # read config
    config = YamlConfig(config_filename)
    sensor_type = config['sensor']['type']
    sensor_frame = config['sensor']['frame']

    # read camera calib
    tf_filename = '%s_to_world.tf' %(sensor_frame)
    T_camera_world = RigidTransform.load(os.path.join(config['calib_dir'], sensor_frame, tf_filename))
    T_camera_world.save(os.path.join(output_dir, tf_filename))

    # setup sensor
    sensor = RgbdSensorFactory.sensor(sensor_type, config['sensor'])

    # start the sensor
    sensor.start()
    color_intrinsics = sensor.color_intrinsics
    ir_intrinsics = sensor.ir_intrinsics
    color_intrinsics.save(os.path.join(output_dir, '%s_color.intr' %(sensor.frame)))    
    ir_intrinsics.save(os.path.join(output_dir, '%s_ir.intr' %(sensor.frame)))

    # get raw images
    for i in range(config['num_images']):
        logging.info('Capturing image %d' %(i))
        color, depth, ir = sensor.frames()
        color.save(os.path.join(output_dir, 'color_%d.png' %(i)))
        depth.save(os.path.join(output_dir, 'depth_%d.npy' %(i)))
        if ir is not None:
            ir.save(os.path.join(output_dir, 'ir_%d.npy' %(i)))

    sensor.stop()
