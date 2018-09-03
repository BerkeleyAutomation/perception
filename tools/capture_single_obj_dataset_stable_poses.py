"""
Script to capture a set of test images.
Be sure to register beforehand!!!
Author: Jeff Mahler
"""
import argparse
import copy
import cv2
import IPython
import logging
import numpy as np
import os
import pcl
import rosgraph.roslogging as rl
import rospy
from scipy import ndimage
import scipy.stats as ss
from skimage import measure
import sys
import trimesh

import matplotlib.pyplot as plt

import autolab_core.utils as utils
from autolab_core import Box, PointCloud, RigidTransform, TensorDataset, YamlConfig
from autolab_core.constants import *
from meshrender import Scene, SceneObject, VirtualCamera, MaterialProperties
from perception import RgbdSensorFactory, Image, RenderMode, BinaryImage, CameraIntrinsics
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d
from dexnet.database import Hdf5Database
from dexnet.constants import READ_ONLY_ACCESS, KEY_SEP_TOKEN

def init_database(cfg):
    database_path = cfg['database_path']
    database = Hdf5Database(database_path, access_level=READ_ONLY_ACCESS)

    obj_keys = []
    for dsname, keys in cfg['object_keys'].iteritems():
        dataset = database.dataset(dsname)
        if keys == 'all':
            cur_obj_keys = dataset.object_keys
        else:
            cur_obj_keys = keys
        for obj_key in cur_obj_keys:
            key = '{}{}{}'.format(dsname, KEY_SEP_TOKEN, obj_key)
            obj_keys.append(key)

    return database, obj_keys

def get_object(cfg, database, key):

    tokens = key.split(KEY_SEP_TOKEN)
    dsname = tokens[0]
    obj_key = tokens[1]
    dataset = database.dataset(dsname)
    obj = dataset[obj_key]
    stable_poses = dataset.stable_poses(obj_key)
    if len(stable_poses) > cfg['max_n_stable_poses']:
        stable_poses = stable_poses[:cfg['max_n_stable_poses']]
    return obj, stable_poses


if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Capture a dataset of depth images from a set of sensors')
    parser.add_argument('output_dir', type=str, help='directory to save output')
    parser.add_argument('--config_filename', type=str, default=None, help='path to configuration file to use')
    args = parser.parse_args()
    output_dir = args.output_dir
    config_filename = args.config_filename

    # make output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # fix config
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/capture_single_obj_dataset_stable_poses.yaml')

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # read config
    config = YamlConfig(config_filename)
    workspace_config = config['workspace']
    camera_config = config['camera']

    # read objects
    objcfg = config['objects']
    database, obj_keys = init_database(objcfg)

    # read workspace bounds
    target_point = (np.array(workspace_config['min_pt']) + np.array(workspace_config['max_pt'])) / 2.0
    target_point[2] = workspace_config['min_pt'][2]

    # setup each sensor
    T_camera_world = RigidTransform.load(os.path.join(camera_config['pose_filename']))
    camera_intr = CameraIntrinsics.load(os.path.join(camera_config['intr_filename']))
    sensor_name = camera_intr.frame
    sensor_dir = os.path.join(output_dir, sensor_name)
    if not os.path.exists(sensor_dir):
        os.makedirs(sensor_dir)
    sensor_tf_filename = os.path.join(sensor_dir, 'T_{}_world.tf'.format(sensor_name))
    sensor_intr_filename = os.path.join(sensor_dir, '{}.intr'.format(sensor_name))
    T_camera_world.save(sensor_tf_filename)
    camera_intr.save(sensor_intr_filename)

    # collect K images
    s = Scene()
    c = VirtualCamera(camera_intr, T_camera_world)
    print T_camera_world
    s.camera = c
    for k in range(len(obj_keys)):
        obj_name = obj_keys[k]
        obj, stable_poses = get_object(objcfg, database, obj_name)
        logging.info('Working on object {}'.format(obj_name))

        # capture
        sobj = SceneObject(obj.mesh, RigidTransform(from_frame='obj', to_frame='world'))
        s.add_object('obj', sobj)

        for j, stp in enumerate(stable_poses):
            logging.info('Test case %d of %d' %(j+1, len(stable_poses)))

            # Put object in stable pose, with (0,0,0) at target min point
            T_obj_world = stp.T_obj_table
            T_obj_world.to_frame='world'
            T_obj_world.translation = target_point
            sobj.T_obj_world = T_obj_world

            # Render target object
            depth_im, segmask = s.wrapped_render([RenderMode.DEPTH, RenderMode.SEGMASK])

            img_dir = os.path.join(sensor_dir, 'color_images')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            depth_im = depth_im.mask_binary(segmask).to_color()
            depth_im.save(os.path.join(img_dir, '{}_{:06d}.png'.format(obj_name, j)))

        s.remove_object('obj')

    del s
