"""
Script to capture a set of test images.
Be sure to register camera beforehand!!!
Author: Jeff Mahler
"""
import argparse
import numpy as np
import os

import rospy
import matplotlib.pyplot as plt

from autolab_core import RigidTransform, Box, YamlConfig, Logger
import autolab_core.utils as utils
from perception import RgbdSensorFactory, Image

# set up logger
logger = Logger.get_logger('tools/capture_test_images.py')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Capture a set of RGB-D images from a set of sensors')
    parser.add_argument('output_dir', type=str, help='path to save captured images')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/capture_test_images.yaml', help='path to configuration file to use')
    args = parser.parse_args()
    config_filename = args.config_filename
    output_dir = args.output_dir

    # read config
    config = YamlConfig(config_filename)
    vis = config['vis']

    # make output dir if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read rescale factor
    rescale_factor = 1.0
    if 'rescale_factor' in config.keys():
        rescale_factor = config['rescale_factor']

    # read workspace bounds
    workspace = None
    if 'workspace' in config.keys():
        workspace = Box(np.array(config['workspace']['min_pt']),
                        np.array(config['workspace']['max_pt']),
                        frame='world')
        
    # init ros node
    rospy.init_node('capture_test_images') #NOTE: this is required by the camera sensor classes
    Logger.reconfigure_root()

    for sensor_name, sensor_config in config['sensors'].iteritems():
        logger.info('Capturing images from sensor %s' %(sensor_name))
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
            logger.info('Capturing image %d' %(i))
            message = 'Hit ENTER when ready.'
            utils.keyboard_input(message=message)
            
            # read images
            color, depth, ir = sensor.frames()

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

                # rescale segmask
                if rescale_factor != 1.0:
                    segmask = segmask.resize(rescale_factor, interp='nearest')
                
                # save segmask
                segmask.save(os.path.join(save_dir, 'segmask_%d.png' %(i)))

            # rescale images
            if rescale_factor != 1.0:
                color = color.resize(rescale_factor)
                depth = depth.resize(rescale_factor, interp='nearest')
            
            # save images
            color.save(os.path.join(save_dir, 'color_%d.png' %(i)))
            depth.save(os.path.join(save_dir, 'depth_%d.npy' %(i)))
            if ir is not None:
                ir.save(os.path.join(save_dir, 'ir_%d.npy' %(i)))
                
            if vis:
                from visualization import Visualizer2D as vis2d
                num_plots = 3 if workspace is not None else 2
                vis2d.figure()
                vis2d.subplot(1,num_plots,1)
                vis2d.imshow(color)
                vis2d.subplot(1,num_plots,2)
                vis2d.imshow(depth)
                if workspace is not None:
                    vis2d.subplot(1,num_plots,3)
                    vis2d.imshow(segmask)
                vis2d.show()
                
        sensor.stop()
