"""
Script to register a webcam to the Photoneo PhoXi
Author: Matt Matl
""" 
import argparse
import logging
import numpy as np
import os
import plyfile
import time

import rospy
import rosgraph.roslogging as rl

from autolab_core import YamlConfig, RigidTransform, PointCloud
from perception import RgbdSensorFactory, ColorImage

from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

def main():
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Register a webcam to the Photoneo PhoXi')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/colorize_phoxi.yaml', help='filename of a YAML configuration for registration')
    args = parser.parse_args()
    config_filename = args.config_filename
    config = YamlConfig(config_filename)

    sensor_data = config['sensors']
    phoxi_config = sensor_data['phoxi']
    phoxi_config['frame'] = 'phoxi'

    # Initialize ROS node
    rospy.init_node('colorize_phoxi', anonymous=True)
    logging.getLogger().addHandler(rl.RosStreamHandler())

    # Get PhoXi sensor set up
    phoxi = RgbdSensorFactory.sensor(phoxi_config['type'], phoxi_config)
    phoxi.start()

    # Capture PhoXi and webcam images
    phoxi_color_im, phoxi_depth_im, _ = phoxi.frames()

    #vis2d.figure()
    #vis2d.subplot(121)
    #vis2d.imshow(phoxi_color_im)
    #vis2d.subplot(122)
    #vis2d.imshow(phoxi_depth_im)
    #vis2d.show()

    phoxi_pc = phoxi.ir_intrinsics.deproject(phoxi_depth_im)
    colors = phoxi_color_im.data.reshape((phoxi_color_im.shape[0] * phoxi_color_im.shape[1], phoxi_color_im.shape[2])) / 255.0
    vis3d.figure()
    vis3d.points(phoxi_pc, color=colors, scale=0.001, subsample=3)
    vis3d.show()

    # Export to PLY file
    vertices = phoxi.ir_intrinsics.deproject(phoxi_depth_im).data.T
    colors = phoxi_color_im.data.reshape(phoxi_color_im.data.shape[0] * phoxi_color_im.data.shape[1], phoxi_color_im.data.shape[2])
    f = open('pcloud.ply', 'w')
    f.write('ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\n'.format(len(vertices)) +
            'property uchar green\nproperty uchar blue\nend_header\n')
    for v, c in zip(vertices,colors):
        f.write('{} {} {} {} {} {}\n'.format(v[0], v[1], v[2], c[0], c[1], c[2]))
    f.close()

if __name__ == '__main__':
    main()

