"""
Try different image filtering methods
Author: Jeff Mahler
"""
import argparse
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import pcl
import os
import rospy
import sys
import time

from autolab_core import Box, PointCloud, RigidTransform
from perception import VirtualSensor
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

min_height = 0.001
max_height = 0.15
rescale_factor = 0.5

vis_clipping = False
vis_segments = False
vis_final_clouds = False
vis_final_images = True

if __name__ == '__main__':
    # set logging
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Filter a set of images')
    parser.add_argument('image_dir', type=str, help='location to read the images from')
    parser.add_argument('frame', type=str, help='frame of images')
    args = parser.parse_args()
    image_dir = args.image_dir
    frame = args.frame

    # sensor
    sensor = VirtualSensor(image_dir)
    sensor.start()
    camera_intr = sensor.ir_intrinsics

    # read tf
    T_camera_world = RigidTransform.load(os.path.join(image_dir, '%s_to_world.tf' %(frame)))
    
    # read images
    color_im, depth_im, _ = sensor.frames()

    # inpaint original image
    depth_im_filtered = depth_im.copy()
    depth_im_orig = depth_im.inpaint(rescale_factor)

    # timing
    filter_start = time.time()
    
    small_depth_im = depth_im.resize(rescale_factor, interp='nearest')
    small_camera_intr = camera_intr.resize(rescale_factor)

    # convert to point cloud in world coords
    deproject_start = time.time()
    point_cloud_cam = small_camera_intr.deproject(small_depth_im)
    point_cloud_cam.remove_zero_points()
    point_cloud_world = T_camera_world * point_cloud_cam

    point_cloud_filtered = copy.deepcopy(point_cloud_world)
    logging.info('Deproject took %.3f sec' %(time.time()-deproject_start))
    
    # filter low
    clip_start = time.time()
    low_indices = np.where(point_cloud_world.data[2,:] < min_height)[0] 
    point_cloud_filtered.data[2,low_indices] = min_height 

    # filter high
    high_indices = np.where(point_cloud_world.data[2,:] > max_height)[0] 
    point_cloud_filtered.data[2,high_indices] = max_height 

    # re-project and update depth im
    #depth_im_filtered = camera_intr.project_to_image(T_camera_world.inverse() * point_cloud_filtered)
    logging.info('Clipping took %.3f sec' %(time.time()-clip_start))
    
    # vis
    focal_point = np.mean(point_cloud_filtered.data,
                          axis=1)
    if vis_clipping:
        vis3d.figure(camera_pose=T_camera_world.as_frames('camera',
                                                          'world'),
                     focal_point=focal_point)
        vis3d.points(point_cloud_world,
                     scale=0.001,
                     color=(1,0,0),
                     subsample=10)
        vis3d.points(point_cloud_filtered,
                     scale=0.001,
                     color=(0,0,1),
                     subsample=10)
        vis3d.show()
    
    pcl_start = time.time()

    # subsample point cloud
    #rate = int(1.0 / rescale_factor)**2
    #point_cloud_filtered = point_cloud_filtered.subsample(rate, random=False)
    box = Box(np.array([0.2, -0.24, min_height]), np.array([0.56, 0.21, max_height]), frame='world')
    point_cloud_masked, valid_indices = point_cloud_filtered.box_mask(box)
    invalid_indices = np.setdiff1d(np.arange(point_cloud_filtered.num_points),
                                   valid_indices)
    
    # apply PCL filters
    pcl_cloud = pcl.PointCloud(point_cloud_masked.data.T.astype(np.float32))
    tree = pcl_cloud.make_kdtree()
    ec = pcl_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.005)
    #ec.set_MinClusterSize(1)
    #ec.set_MaxClusterSize(250)
    ec.set_MinClusterSize(250)
    ec.set_MaxClusterSize(1000000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    num_clusters = len(cluster_indices)

    segments = []
    filtered_points = np.zeros([3,point_cloud_masked.num_points])
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
        
        seg_point_cloud = PointCloud(points, frame='world')
        segments.append(seg_point_cloud)

    all_points = np.c_[filtered_points[:,:cur_i], point_cloud_filtered.data[:,invalid_indices]]
    point_cloud_filtered = PointCloud(all_points, frame='world')        
    pcl_stop = time.time()
    logging.info('PCL Seg took %.3f sec' %(pcl_stop-pcl_start))
        
    if vis_segments:
        vis3d.figure(camera_pose=T_camera_world.as_frames('camera',
                                                          'world'),
                     focal_point=focal_point)
        for i, segment in enumerate(segments):
            color = plt.get_cmap('hsv')(float(i)/num_clusters)[:-1]
            vis3d.points(segment,
                         scale=0.001,
                         color=color,
                         subsample=5)
        vis3d.show()

    if vis_final_clouds:
        vis3d.figure(camera_pose=T_camera_world.as_frames('camera',
                                                          'world'),
                     focal_point=focal_point)
        #vis3d.points(point_cloud_world,
        #             scale=0.001,
        #             color=(1,0,0),
        #             subsample=10)
        vis3d.points(point_cloud_filtered,
                     scale=0.001,
                     color=(0,0,1),
                     subsample=5)
        vis3d.show()
        
    # convert to depth image
    project_start = time.time()
    point_cloud_cam = T_camera_world.inverse() * point_cloud_filtered
    #depth_im_noise = small_camera_intr.project_to_image(point_cloud_cam)
    depth_im_filtered = small_camera_intr.project_to_image(point_cloud_cam)    
    #depth_im_filtered = depth_im_filtered.resize(1.0/rescale_factor)#, interp='nearest')
    noise_mask = depth_im_filtered.to_binary()
    #depth_im_filtered = depth_im_filtered.inpaint(rescale_factor)
    #depth_im_noise = depth_im_noise.resize(1.0/rescale_factor)
    #noise_mask = depth_im_noise.to_binary()
    logging.info('Project took %.3f sec' %(time.time() - project_start))
    #depth_im_filtered = depth_im.mask_binary(noise_mask)
    #depth_im_filtered = depth_im_filtered.mask_binary(noise_mask.inverse())
    depth_im_filtered = depth_im_filtered.inpaint(0.5)
    
    filter_stop = time.time()
    logging.info('Filtering took %.3f sec' %(filter_stop-filter_start))
    
    if vis_final_images:
        vis2d.figure()
        vis2d.subplot(2,2,1)
        vis2d.imshow(depth_im)
        vis2d.title('Orig')
        vis2d.subplot(2,2,2)
        vis2d.imshow(depth_im_orig)
        vis2d.title('Inpainted')
        vis2d.subplot(2,2,3)
        vis2d.imshow(noise_mask)
        vis2d.title('Mask')
        vis2d.subplot(2,2,4)
        vis2d.imshow(depth_im_filtered)
        vis2d.title('Filtered')
        vis2d.show()
