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

def render_workspace_image(camera_intr, T_camera_world):
    pass

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

    # read workspace bounds
    workspace = None
    if 'workspace' in config.keys():
        workspace = Box(np.array(config['workspace']['min_pt']),
                        np.array(config['workspace']['max_pt']),
                        frame='world')
        
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
        camera_intr = sensor.ir_intrinsics
        camera_intr.save(os.path.join(save_dir, '%s.intr' %(sensor.frame)))

        # get raw images
        for i in range(sensor_config['num_images']):
            logging.info('Capturing image %d' %(i))

            # save raw images
            color, depth, ir = sensor.frames()
            color.save(os.path.join(save_dir, 'color_%d.png' %(i)))
            depth.save(os.path.join(save_dir, 'depth_%d.npy' %(i)))
            if ir is not None:
                ir.save(os.path.join(save_dir, 'ir_%d.npy' %(i)))

            # save processed images
            if workspace is not None:
                # deproject into 3D world coordinates
                point_cloud_cam = camera_intr.deproject(depth)
                point_cloud_cam.remove_zero_points()
                point_cloud_world = T_camera_world * point_cloud_cam
                
                # segment out the region in the workspace (objects only)
                seg_point_cloud_world, _ = point_cloud_world.box_mask(workspace)

                # compute the segmask for points above the box
                seg_point_cloud_cam = T_camera_world.inverse() * seg_point_cloud_world
                depth_im_seg = camera_intr.project_to_image(seg_point_cloud_cam)
                segmask = depth_im_seg.to_binary()

                from visualization import Visualizer2D as vis2d
                vis2d.figure()
                vis2d.subplot(1,3,1)
                vis2d.imshow(color)
                vis2d.subplot(1,3,2)
                vis2d.imshow(depth)
                vis2d.subplot(1,3,3)
                vis2d.imshow(segmask)
                vis2d.show()
                
        sensor.stop()
