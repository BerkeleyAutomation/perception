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

from autolab_core import PointCloud, RigidTransform
from perception import VirtualSensor
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

min_height = 0.001
max_height = 0.15
rescale_factor = 0.25

vis_clipping = False
vis_segments = False
vis_final_clouds = True
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
    depth_im_orig = depth_im.inpaint(rescale_factor)
    
    # convert to point cloud in world coords
    #small_depth_im = depth_im.resize(rescale_factor)
    #small_camera_intr = camera_intr.resize(rescale_factor)

    point_cloud_cam = camera_intr.deproject(depth_im)
    point_cloud_cam.remove_zero_points()
    point_cloud_world = T_camera_world * point_cloud_cam

    point_cloud_filtered = copy.deepcopy(point_cloud_world)

    # filter low
    low_indices = np.where(point_cloud_world.data[2,:] < min_height)[0] 
    point_cloud_filtered.data[2,low_indices] = min_height 

    # filter high
    high_indices = np.where(point_cloud_world.data[2,:] > max_height)[0] 
    point_cloud_filtered.data[2,high_indices] = max_height 

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
    
    # PCL filters
    pcl_start = time.time()
    pcl_cloud = pcl.PointCloud(point_cloud_filtered.data.T.astype(np.float32))
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
    filtered_points = np.zeros([3,point_cloud_filtered.num_points])
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

    
    point_cloud_filtered = PointCloud(filtered_points[:,:cur_i], frame='world')        
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
    point_cloud_cam = T_camera_world.inverse() * point_cloud_filtered
    depth_im_filtered = camera_intr.project_to_image(point_cloud_cam)
    depth_im_filtered = depth_im_filtered.inpaint(rescale_factor)
    #depth_im_filtered = depth_im_filtered.resize(1.0/rescale_factor)
    
    if vis_final_images:
        vis2d.figure()
        vis2d.subplot(1,3,1)
        vis2d.imshow(depth_im)
        vis2d.title('Orig')
        vis2d.subplot(1,3,2)
        vis2d.imshow(depth_im_orig)
        vis2d.title('Inpainted')
        vis2d.subplot(1,3,3)
        vis2d.imshow(depth_im_filtered)
        vis2d.title('Filtered')
        vis2d.show()
