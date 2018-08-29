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
from perception import RgbdSensorFactory, Image, RenderMode, BinaryImage
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

GUI_PAUSE = 0.5

def preprocess_images(raw_color_im,
                      raw_depth_im,
                      camera_intr,
                      T_camera_world,
                      workspace_box,
                      workspace_im,
                      image_proc_config):
    """ Preprocess a set of color and depth images. """
    # read params
    inpaint_rescale_factor = image_proc_config['inpaint_rescale_factor']
    cluster = image_proc_config['cluster']
    cluster_tolerance = image_proc_config['cluster_tolerance']
    min_cluster_size = image_proc_config['min_cluster_size']
    max_cluster_size = image_proc_config['max_cluster_size']

    # deproject into 3D world coordinates
    point_cloud_cam = camera_intr.deproject(raw_depth_im)
    point_cloud_cam.remove_zero_points()
    point_cloud_world = T_camera_world * point_cloud_cam

    # compute the segmask for points above the box
    seg_point_cloud_world, _ = point_cloud_world.box_mask(workspace_box)
    seg_point_cloud_cam = T_camera_world.inverse() * seg_point_cloud_world
    depth_im_seg = camera_intr.project_to_image(seg_point_cloud_cam)

    # mask out objects in the known workspace
    env_pixels = depth_im_seg.pixels_farther_than(workspace_im)
    depth_im_seg._data[env_pixels[:,0], env_pixels[:,1]] = 0

    # REMOVE NOISE
    # clip low points
    low_indices = np.where(point_cloud_world.data[2,:] < workspace_box.min_pt[2])[0] 
    point_cloud_world.data[2,low_indices] = workspace_box.min_pt[2]
            
    # clip high points
    high_indices = np.where(point_cloud_world.data[2,:] > workspace_box.max_pt[2])[0] 
    point_cloud_world.data[2,high_indices] = workspace_box.max_pt[2]

    # segment out the region in the workspace (including the table)
    workspace_point_cloud_world, valid_indices = point_cloud_world.box_mask(workspace_box)
    invalid_indices = np.setdiff1d(np.arange(point_cloud_world.num_points),
                                   valid_indices)

    if cluster:
        # create new cloud
        pcl_cloud = pcl.PointCloud(workspace_point_cloud_world.data.T.astype(np.float32))
        tree = pcl_cloud.make_kdtree()
    
        # find large clusters (likely to be real objects instead of noise)
        ec = pcl_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(cluster_tolerance)
        ec.set_MinClusterSize(min_cluster_size)
        ec.set_MaxClusterSize(max_cluster_size)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()
        num_clusters = len(cluster_indices)

        # read out all points in large clusters
        filtered_points = np.zeros([3,workspace_point_cloud_world.num_points])
        cur_i = 0
        for j, indices in enumerate(cluster_indices):
            num_points = len(indices)
            points = np.zeros([3,num_points])
    
            for i, index in enumerate(indices):
                points[0,i] = pcl_cloud[index][0]
                points[1,i] = pcl_cloud[index][1]
                points[2,i] = pcl_cloud[index][2]

            filtered_points[:,cur_i:cur_i+num_points] = points.copy()
            cur_i = cur_i + num_points

        # reconstruct the point cloud
        all_points = np.c_[filtered_points[:,:cur_i], point_cloud_world.data[:,invalid_indices]]
    else:
        all_points = point_cloud_world.data
    filtered_point_cloud_world = PointCloud(all_points,
                                            frame='world')  

    # compute the filtered depth image
    filtered_point_cloud_cam = T_camera_world.inverse() * filtered_point_cloud_world
    depth_im = camera_intr.project_to_image(filtered_point_cloud_cam)    

    # form segmask
    segmask = depth_im_seg.to_binary()
    valid_px_segmask = depth_im.invalid_pixel_mask().inverse()
    segmask = segmask.mask_binary(valid_px_segmask)
    segdata = segmask.data

    # Remove any tiny cc's
    cc_labels = measure.label(segdata)
    num_ccs = np.max(cc_labels)
    for i in range(1, num_ccs + 1):
        cc_mask = (cc_labels == i)
        cc_size = np.count_nonzero(cc_mask)
        if cc_size < 30:
            segdata[np.where(cc_mask)] = 0

    #segdata = cv2.erode(segdata, np.ones((10,10), np.uint8), iterations=1)
    #segdata = cv2.dilate(segdata, np.ones((10,10), np.uint8), iterations=1)
    segmask = BinaryImage(ndimage.binary_fill_holes(segdata).astype(np.uint8) * 255)
    region_segdata = np.zeros(segmask.data.shape, dtype=np.uint8)
    region_segdata[150:segmask.data.shape[0]-150,150:segmask.data.shape[1]-150] = 255
    segmask = segmask.mask_binary(BinaryImage(region_segdata))

    # inpaint
    color_im = raw_color_im.inpaint(rescale_factor=inpaint_rescale_factor)
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)    
    return color_im, depth_im, segmask    
    
if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)
    rospy.init_node('capture_dataset', anonymous=True)
    logging.getLogger().addHandler(rl.RosStreamHandler())

    # parse args
    parser = argparse.ArgumentParser(description='Capture a dataset of RGB-D images from a set of sensors')
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
                                       'cfg/tools/capture_single_obj_dataset.yaml')

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)


    # read config
    config = YamlConfig(config_filename)
    sensor_configs = config['sensors']
    workspace_config = config['workspace']
    image_proc_config = config['image_proc']

    # read objects
    objects = config['objects']
    num_objects = len(objects)
    n_samples_per_object = config['n_samples_per_object']
    im_rescale_factor = image_proc_config['im_rescale_factor']

    save_raw = config['save_raw']
    vis = config['vis']

    # open gui
    gui = plt.figure(0, figsize=(8,8))
    plt.ion()
    plt.title('INITIALIZING')
    plt.imshow(np.zeros([100,100]),
               cmap=plt.cm.gray_r)
    plt.axis('off')
    plt.draw()
    plt.pause(GUI_PAUSE)
    
    # read workspace bounds
    workspace_box = Box(np.array(workspace_config['min_pt']),
                        np.array(workspace_config['max_pt']),
                        frame='world')
    
    # read workspace objects
    workspace_objects = {}
    for obj_key, obj_config in workspace_config['objects'].iteritems():
        mesh_filename = obj_config['mesh_filename']
        pose_filename = obj_config['pose_filename']
        print(mesh_filename)
        obj_mesh = trimesh.load_mesh(mesh_filename)
        obj_pose = RigidTransform.load(pose_filename)
        obj_mat_props = MaterialProperties(smooth=True,
                                           wireframe=False)
        scene_obj = SceneObject(obj_mesh, obj_pose, obj_mat_props)
        workspace_objects[obj_key] = scene_obj

    # setup each sensor
    datasets = {}
    sensors = {}
    sensor_poses = {}
    camera_intrs = {}
    workspace_ims = {}
    for sensor_name, sensor_config in sensor_configs.iteritems():
        # read params
        sensor_type = sensor_config['type']
        sensor_frame = sensor_config['frame']

        sensor_dir = os.path.join(output_dir, sensor_name)
        if not os.path.exists(sensor_dir):
            os.makedirs(sensor_dir)

        # read camera calib
        tf_filename = '%s_to_world.tf' %(sensor_frame)
        T_camera_world = RigidTransform.load(os.path.join(sensor_config['calib_dir'], sensor_frame, tf_filename))
        sensor_poses[sensor_name] = T_camera_world

        # setup sensor
        sensor = RgbdSensorFactory.sensor(sensor_type, sensor_config)
        sensors[sensor_name] = sensor

        # start the sensor
        sensor.start()
        camera_intr = sensor.ir_intrinsics
        camera_intr = camera_intr.resize(im_rescale_factor)
        camera_intrs[sensor_name] = camera_intr

        # render image of static workspace
        scene = Scene()
        camera = VirtualCamera(camera_intr, T_camera_world)
        scene.camera = camera
        for obj_key, scene_obj in workspace_objects.iteritems():
            scene.add_object(obj_key, scene_obj)
        workspace_ims[sensor_name] = scene.wrapped_render([RenderMode.DEPTH])[0]

        # save intrinsics and pose
        sensor_tf_filename = os.path.join(sensor_dir, 'T_{}_world.tf'.format(sensor_name))
        sensor_intr_filename = os.path.join(sensor_dir, '{}.intr'.format(sensor_name))
        T_camera_world.save(sensor_tf_filename)
        camera_intr.save(sensor_intr_filename)

        # save raw
        if save_raw:
            sensor_dir = os.path.join(output_dir, sensor_name)
            raw_dir = os.path.join(sensor_dir, 'raw')
            if not os.path.exists(raw_dir):
                os.mkdir(raw_dir)

            camera_intr_filename = os.path.join(raw_dir, 'camera_intr.intr')
            camera_intr.save(camera_intr_filename)
            camera_pose_filename = os.path.join(raw_dir, 'T_camera_world.tf')
            T_camera_world.save(camera_pose_filename)

    # collect K images
    for k in range(num_objects):
        obj_name = objects[k]
        logging.info('Working on object {}'.format(obj_name))

        for j in range(n_samples_per_object):
            logging.info('Test case %d of %d' %(j+1, n_samples_per_object))

            # get human consent
            message = 'Please place object: {}\n'.format(obj_name)
            message += 'Hit ENTER when finished.'
            utils.keyboard_input(message=message)

            # capture
            for sensor_name, sensor in sensors.iteritems():
                logging.info('Capturing images from sensor %s' %(sensor_name))

                # read pose and intrinsics
                sensor_pose = sensor_poses[sensor_name]
                camera_intr = camera_intrs[sensor_name]
                workspace_im = workspace_ims[sensor_name]
                T_camera_world = sensor_pose

                # read raw images
                raw_color_im, raw_depth_im, _ = sensor.frames()

                # resize
                raw_color_im = raw_color_im.resize(im_rescale_factor)
                raw_depth_im = raw_depth_im.resize(im_rescale_factor,
                                                interp='nearest')

                # preprocess
                color_im, depth_im, segmask = preprocess_images(raw_color_im,
                                                                raw_depth_im,
                                                                camera_intr,
                                                                T_camera_world,
                                                                workspace_box,
                                                                workspace_im,
                                                                image_proc_config)

                # visualize
                if vis:
                    gui = plt.figure(0)
                    plt.clf()
                    vis2d.subplot(2,3,1)
                    vis2d.imshow(raw_color_im)
                    vis2d.title('RAW COLOR')
                    vis2d.subplot(2,3,2)
                    vis2d.imshow(raw_depth_im)
                    vis2d.title('RAW DEPTH')
                    vis2d.subplot(2,3,4)
                    vis2d.imshow(color_im)
                    vis2d.title('COLOR')
                    vis2d.subplot(2,3,5)
                    vis2d.imshow(depth_im)
                    vis2d.title('DEPTH')
                    vis2d.subplot(2,3,6)
                    vis2d.imshow(segmask)
                    vis2d.title('SEGMASK')
                    plt.draw()
                    plt.pause(GUI_PAUSE)


                sensor_dir = os.path.join(output_dir, sensor_name)

                img_dir = os.path.join(sensor_dir, 'color_images')
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                color_im = color_im.mask_binary(segmask)
                color_im.save(os.path.join(img_dir, '{}_{:06d}.png'.format(obj_name, j)))

                img_dir = os.path.join(sensor_dir, 'depth_images')
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                depth_im = depth_im.mask_binary(segmask)
                np.save(os.path.join(img_dir, '{}_{:06d}.npy'.format(obj_name, j)), depth_im.data)

    # stop all sensors
    for sensor_name, sensor in sensors.iteritems():
        sensor.stop()
