"""
Test to compare normal computation methods
Author: Jeff Mahler
"""
import cv2
import numpy as np
import pcl
import os
import sys

import IPython

from autolab_core import RigidTransform
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

from perception import DepthImage, CameraIntrinsics

KSIZE = 9

if __name__ == '__main__':
    depth_im_filename = sys.argv[1]
    camera_intr_filename = sys.argv[2]

    camera_intr = CameraIntrinsics.load(camera_intr_filename)
    depth_im = DepthImage.open(depth_im_filename, frame=camera_intr.frame)

    depth_im = depth_im.inpaint()
    
    point_cloud_im = camera_intr.deproject_to_image(depth_im)
    normal_cloud_im = point_cloud_im.normal_cloud_im(ksize=KSIZE)

    vis3d.figure()
    vis3d.points(point_cloud_im.to_point_cloud(), scale=0.0025)

    alpha = 0.025
    subsample = 20
    for i in range(0, point_cloud_im.height, subsample):
        for j in range(0, point_cloud_im.width, subsample):
            p = point_cloud_im[i,j]
            n = normal_cloud_im[i,j]
            n2 = normal_cloud_im_s[i,j]
            if np.linalg.norm(n) > 0:
                points = np.array([p, p + alpha*n])
                vis3d.plot3d(points, tube_radius=0.001, color=(1,0,0))

                points = np.array([p, p + alpha*n2])
                vis3d.plot3d(points, tube_radius=0.001, color=(1,0,1))

    vis3d.show()
