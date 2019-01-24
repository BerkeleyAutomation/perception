"""
Script to capture a set of test images.
Be sure to register beforehand!!!
Author: Jeff Mahler
"""
import argparse
import copy
import IPython
import logging
import numpy as np
import os
import pcl
import rosgraph.roslogging as rl
import rospy
import scipy.stats as ss
import sys
import trimesh

import matplotlib.pyplot as plt

import autolab_core.utils as utils
from autolab_core import Box, PointCloud, RigidTransform, TensorDataset, YamlConfig
from autolab_core.constants import *
from meshrender import Scene, SceneObject, VirtualCamera, MaterialProperties
from perception import RgbdSensorFactory, Image, RenderMode
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
    parser.add_argument('num_images', type=int, help='number of images to capture')
    parser.add_argument('--config_filename', type=str, default=None, help='path to configuration file to use')
    args = parser.parse_args()
    output_dir = args.output_dir
    num_images = args.num_images
    config_filename = args.config_filename

    # make output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # fix config
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/capture_dataset.yaml')

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    
    # read config
    config = YamlConfig(config_filename)
    dataset_config = config['dataset']
    sensor_configs = config['sensors']
    workspace_config = config['workspace']
    image_proc_config = config['image_proc']

    # read objects
    train_pct = config['train_pct']
    objects = config['objects']
    num_objects = len(objects)
    num_train = int(np.ceil(train_pct * num_objects))
    num_test = num_objects - num_train
    all_indices = np.arange(num_objects)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:num_train]
    test_indices = all_indices[num_train:]

    num_train_images = int(np.ceil(train_pct * num_images))
    all_image_indices = np.arange(num_images)
    np.random.shuffle(all_image_indices)
    train_image_indices = all_image_indices[:num_train_images]

    # set random variable for the number of objects
    mean_num_objects = config['mean_num_objects']
    min_num_objects = config['min_num_objects']
    max_num_objects = config['max_num_objects']
    num_objs_rv = ss.poisson(mean_num_objects-1)
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

        # fix dataset config
        dataset_config['fields']['raw_color_ims']['height'] = camera_intr.height
        dataset_config['fields']['raw_color_ims']['width'] = camera_intr.width
        dataset_config['fields']['raw_depth_ims']['height'] = camera_intr.height
        dataset_config['fields']['raw_depth_ims']['width'] = camera_intr.width 
        dataset_config['fields']['color_ims']['height'] = camera_intr.height
        dataset_config['fields']['color_ims']['width'] = camera_intr.width 
        dataset_config['fields']['depth_ims']['height'] = camera_intr.height
        dataset_config['fields']['depth_ims']['width'] = camera_intr.width 
        dataset_config['fields']['segmasks']['height'] = camera_intr.height
        dataset_config['fields']['segmasks']['width'] = camera_intr.width 
       
        # open dataset
        sensor_dataset_filename = os.path.join(output_dir, sensor_name)
        datasets[sensor_name] = TensorDataset(sensor_dataset_filename,
                                              dataset_config)        

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
    for k in range(num_images):
        logging.info('Test case %d of %d' %(k, num_images))

        # set test case
        train = 0
        split = TEST_ID
        if k in train_image_indices:
            train = 1
            split = TRAIN_ID
        if train:
            num_objects = min(max(num_objs_rv.rvs(size=1)[0] + 1, min_num_objects), num_train)
            obj_names = [objects[i] for i in np.random.choice(train_indices, size=num_objects, replace=False)]
        else:
            num_objects = min(max(num_objs_rv.rvs(size=1)[0] + 1, min_num_objects), num_test)
            obj_names = [objects[i] for i in np.random.choice(test_indices, size=num_objects, replace=False)]
            
        # get human consent
        message = 'Please place %d objects:\n' %(num_objects)
        for name in obj_names:
            message += '\t{}\n'.format(name)
        message += 'Hit ENTER when finished.'
        utils.keyboard_input(message=message)

        # capture
        for sensor_name, sensor in sensors.iteritems():
            logging.info('Capturing images from sensor %s' %(sensor_name))

            # read pose and intrinsics
            sensor_pose = sensor_poses[sensor_name]
            camera_intr = camera_intrs[sensor_name]
            workspace_im = workspace_ims[sensor_name]
            dataset = datasets[sensor_name]
            T_camera_world = sensor_pose
            datapoint = dataset.datapoint_template
            
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
            
            # save data
            datapoint['split'] = split
            datapoint['camera_intrs'] = camera_intr.vec
            datapoint['camera_poses'] = sensor_pose.vec
            datapoint['raw_color_ims'] = raw_color_im.raw_data
            datapoint['raw_depth_ims'] = raw_depth_im.raw_data
            datapoint['color_ims'] = color_im.raw_data
            datapoint['depth_ims'] = depth_im.raw_data
            datapoint['segmasks'] = segmask.raw_data
            dataset.add(datapoint)

            # save raw data
            if save_raw:
                sensor_dir = os.path.join(output_dir, sensor_name)
                raw_dir = os.path.join(sensor_dir, 'raw')

                raw_color_im_filename = os.path.join(raw_dir, 'raw_color_%d.png' %(k))
                raw_color_im.save(raw_color_im_filename)
                color_im_filename = os.path.join(raw_dir, 'color_%d.png' %(k))
                color_im.save(color_im_filename)
                
                raw_depth_im_filename = os.path.join(raw_dir, 'raw_depth_%d.npy' %(k))
                raw_depth_im.save(raw_depth_im_filename)
                depth_im_filename = os.path.join(raw_dir, 'depth_%d.npy' %(k))
                depth_im.save(depth_im_filename)

                segmask_filename = os.path.join(raw_dir, 'segmask_%d.png' %(k))
                segmask.save(segmask_filename)
                
    # stop all sensors
    for sensor_name, sensor in sensors.iteritems():
        datasets[sensor_name].flush()
        sensor.stop()
