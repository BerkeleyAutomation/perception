'''
Process kinect image in preparation for querying database for registration
Authors: Jeff, Jacky
'''
import os
import argparse
import numpy as np
import logging
import yaml
import scipy.ndimage.filters as skf

import matplotlib.pyplot as plt

import IPython

from image import Image, DepthImage
from alan.core import RigidTransform, Box
from alan.core import Visualizer as vis

class DepthToCNNQueryImage:

    def __init__(self, T_camera_world, ir_intrinsics, depth_im_median_filter_dim, 
                area_thresh, index_im_dim, num_areas=1):

        self.T_camera_world = T_camera_world
        self.ir_intrinsics = ir_intrinsics
        self.depth_im_median_filter_dim = depth_im_median_filter_dim        
        self.area_thresh = area_thresh        
        self.index_im_dim = index_im_dim
        self.num_areas = num_areas

    def isolate_workspace(self, point_cloud_camera, depth_img):
        # TODO: specify actual corners of masking box
        corner1 = np.array([-0.12, -0.5, 0.01])
        corner2 = np.array([0.5, 0.5, 0.25])
        height, width, _ = depth_img.shape

        # change point cloud frame to world frame
        point_cloud_world = self.T_camera_world * point_cloud_camera

        # threshold to find objects on the table
        box = Box(corner1, corner2, 'world')
        point_cloud_world_isolated, point_cloud_world_isolated_ind = point_cloud_world.box_mask(box)
        depth_img_masked = depth_img.filter_by_linear_ind(point_cloud_world_isolated_ind)

        '''
        T_origin_world = RigidTransform(from_frame='origin')
        T_corner1_world = RigidTransform(translation=corner1, from_frame='corner1')
        T_corner2_world = RigidTransform(translation=corner2, from_frame='corner2')

        vis.figure()
        vis.points(point_cloud_world, color=(1,0,0), subsample_rate=20, scale=0.005)
        vis.plot_pose(T_camera_world)
        vis.plot_pose(T_origin_world)
        vis.plot_pose(T_corner1_world)
        vis.plot_pose(T_corner2_world)
        vis.show()
        '''
        return depth_img_masked

    def filter_binary_image(self, binary_img):
        # median filter
        binary_img_median = binary_img.apply(skf.median_filter, size=self.depth_im_median_filter_dim)

        # keep largest connected object
        binary_img_pruned = binary_img_median.prune_contours(area_thresh=self.area_thresh, num_areas=self.num_areas)

        return binary_img_pruned

    def get_query_image(self, depth_images, output_path=None):
        '''
        Creates and returns a query image. 
        If output path is given, intermediate images will be saved for
        debug purposes to the output path.
        '''
        logging.info("Getting median of depth images.")
        depth_image = Image.median_images(depth_images)

        logging.info("Deprojecting into point clouds.")
        raw_point_cloud = self.ir_intrinsics.deproject(depth_image)

        # isolating workspace through point clouds
        logging.info("Isolating workspace in point clouds.")
        depth_img_masked = self.isolate_workspace(raw_point_cloud, depth_image)

        # turn depth into binary image
        logging.info("Binarizing depth image.")
        binary_img_masked = depth_img_masked.to_binary()
        
        # filter binary image by pruning contours and applying median filter
        logging.info("Filtering binary image.")
        binary_img_filtered = self.filter_binary_image(binary_img_masked)
        
        if binary_img_filtered is None:
            logging.warn("No sizeable object detected in scene! Returning None.")
            return None

        # centering filtered binary image
        logging.info("Centering binary image.")
        binary_img_centered, diff_px = binary_img_filtered.center_nonzero(self.ir_intrinsics)

        nonzeros = np.where(binary_img_filtered.data != 0)
        target_i = nonzeros[0][0]
        target_j = nonzeros[1][0]

        T_im_im_centered = depth_img_masked.get_rigid_transform_px_shift(self.ir_intrinsics,
                                                                    np.array([target_i, target_j]), 
                                                                    diff_px, binary_img_centered.frame)

        # cropping centered binary image
        logging.info("Cropping binary image.")
        binary_img_cropped = binary_img_centered.crop(self.index_im_dim, self.index_im_dim)

        if output_path is not None:
            logging.info("Saving images.")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            depth_image.savefig(output_path, '0 median depth image', cmap=plt.cm.gray)
            depth_img_masked.savefig(output_path, '1 masked depth image', cmap=plt.cm.gray)
            binary_img_masked.savefig(output_path, '2 masked binary image', cmap=plt.cm.gray)
            binary_img_filtered.savefig(output_path, '3 filtered binary image', cmap=plt.cm.gray)
            binary_img_centered.savefig(output_path, '4 center binary image', cmap=plt.cm.gray)
            binary_img_cropped.savefig(output_path, '5 cropped binary image', cmap=plt.cm.gray)
            
        return binary_img_cropped, T_im_im_centered