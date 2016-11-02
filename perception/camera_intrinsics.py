"""
Encapsulates camera intrinsic parameters for projecting / deprojecitng points
Author: Jeff Mahler
"""
import copy
import IPython
import numpy as np
import json
import os

from image import DepthImage, PointCloudImage
from core import Point, PointCloud, ImageCoords

from constants import INTR_EXTENSION

class CameraIntrinsics(object):
    def __init__(self, frame, fx, fy=None, cx=0.0, cy=0.0, skew=0.0, height=None, width=None):
        """
        Init camera parameters
        
        Params:
           frame: (string) the frame of reference for the point cloud
           fx: (float) x focal length of camera in pixels
           fy: (float) y focal length of camera in pixels
           cx: (float) optical center of camera in pixels along x axis
           cy: (float) optical center of camera in pixels along y axis
           skew: (float) skew of camera in pixels
           height: (float) height of camera in pixels
           width: (float) width of camera image in pixels
        """
        self._frame = frame
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._skew = skew
        self._height = height
        self._width = width

        # set focal, camera center automatically if under specified
        if fy is None:
            self._fy = fx

        # set camera projection matrix
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [       0,   self._fy, self._cy],
                            [       0,          0,        1]])

    @property
    def frame(self):
        return self._frame

    @property
    def fx(self):
        return self._fx

    @property
    def fy(self):
        return self._fy

    @property
    def cx(self):
        return self._cx

    @cx.setter
    def cx(self, z):
        self._cx = z
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [       0,   self._fy, self._cy],
                            [       0,          0,        1]])

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, z):
        self._cy = z
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [       0,   self._fy, self._cy],
                            [       0,          0,        1]])
    @property
    def skew(self):
        return self._skew

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def proj_matrix(self):
        return self.K
    
    @property
    def K(self):
        return self._K

    def project(self, point_cloud, round_px=True):
        """
        Projects a point cloud into the camera given by these parameters
                
        Params:
           point_cloud: (PointCloud or Point object) point cloud of N 3D points to project
           round_px: (bool) whether or not to round to the nearest pixels
        Returns:
           ImageCoords object - acts like a float array of 2D image coordinates
        """
        # check valid data
        if not isinstance(point_cloud, PointCloud) and not (isinstance(point_cloud, Point) and point_cloud.dim == 3):
            raise ValueError('Must provide PointCloud or 3D Point object for projection')
        if point_cloud.frame != self._frame:
            raise ValueError('Cannot project points in frame %s into camera with frame %s' %(point_cloud.frame, self._frame))
        
        points_proj = self._K.dot(point_cloud.data)
        if len(points_proj.shape) == 1:
            points_proj = points_proj[:, np.newaxis]
        point_depths = np.tile(points_proj[2,:], [3, 1])
        points_proj = np.divide(points_proj, point_depths)
        if round_px:
            points_proj = np.round(points_proj)
                    
        if isinstance(point_cloud, Point):
            return Point(data=points_proj[:2,:].astype(np.int16), frame=self._frame)
        return ImageCoords(data=points_proj[:2,:].astype(np.int16), frame=self._frame)

    def project_to_image(self, point_cloud, round_px=True):
        """
        Projects a point cloud into the camera given by these parameters
                
        Params:
           point_cloud: (PointCloud or Point object) point cloud of N 3D points to project
           round_px: (bool) whether or not to round to the nearest pixels
        Returns:
           DepthImage object
        """
        # check valid data
        if not isinstance(point_cloud, PointCloud) and not (isinstance(point_cloud, Point) and point_cloud.dim == 3):
            raise ValueError('Must provide PointCloud or 3D Point object for projection')
        if point_cloud.frame != self._frame:
            raise ValueError('Cannot project points in frame %s into camera with frame %s' %(point_cloud.frame, self._frame))
        
        points_proj = self._K.dot(point_cloud.data)
        if len(points_proj.shape) == 1:
            points_proj = points_proj[:, np.newaxis]
        point_depths = points_proj[2,:]
        point_z = np.tile(point_depths, [3, 1])
        points_proj = np.divide(points_proj, point_z)
        if round_px:
            points_proj = np.round(points_proj)
        points_proj = points_proj[:2,:].astype(np.int16)

        valid_ind = np.where((points_proj[0,:] >= 0) & \
                             (points_proj[1,:] >= 0) & \
                             (points_proj[0,:] < self.width) & \
                             (points_proj[1,:] < self.height))[0]

        depth_data = np.zeros([self.height, self.width])
        depth_data[points_proj[1,valid_ind], points_proj[0,valid_ind]] = point_depths[valid_ind]
        return DepthImage(depth_data, frame=self.frame)

    def deproject(self, depth_image):
        """
        Deprojects a depth image into a point cloud

        Params:
           depth_image: (DepthImage object) 2D depth image to project
        Returns:
           PointCloud object
        """
        # check valid input
        if not isinstance(depth_image, DepthImage):
            raise ValueError('Must provide DepthImage object for projection')
        if depth_image.frame != self._frame:
            raise ValueError('Cannot deproject points in frame %s from camera with frame %s' %(depth_image.frame, self._frame))

        # create homogeneous pixels 
        row_indices = np.arange(depth_image.height)
        col_indices = np.arange(depth_image.width)
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_image.data.flatten(), [3,1])
            
        # deproject
        points_3d = depth_arr * np.linalg.inv(self._K).dot(pixels_homog)
        return PointCloud(data=points_3d, frame=self._frame)

    def deproject_to_image(self, depth_image):
        """
        Deprojects a depth image into a point cloud image

        Params:
           depth_image: (DepthImage object) 2D depth image to project
        Returns:
           PointCloudImage object
        """
        point_cloud = self.deproject(depth_image)
        point_cloud_im_data = point_cloud.data.T.reshape(depth_image.height, depth_image.width, 3)
        return PointCloudImage(data=point_cloud_im_data,
                               frame=self._frame)

    def deproject_pixel(self, depth, pixel):
        """
        Deprojects a single pixel with depth d to a Point

        Params:
           depth: (float) depth value at pixel px
           pixel: (2D Point object) pixel to project
        Returns:
           Point object
        """
        # check valid input
        if not isinstance(pixel, Point) and not pixel.dim == 2:
            raise ValueError('Must provide 2D Point object for pixel projection')
        if pixel.frame != self._frame:
            raise ValueError('Cannot deproject pixel in frame %s from camera with frame %s' %(pixel.frame, self._frame))

        # deproject
        point_3d = depth * np.linalg.inv(self._K).dot(np.r_[pixel.data, 1.0])
        return Point(data=point_3d, frame=self._frame)        
    
    def save(self, filename):
        """ Save the intrinsics to file """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() != INTR_EXTENSION:
            raise ValueError('Extension %s not supported for CameraIntrinsics. Must be stored with extension %s' %(file_ext, INTR_EXTENSION))

        camera_intr_dict = copy.deepcopy(self.__dict__)
        camera_intr_dict['_K'] = 0 # can't save matrix
        f = open(filename, 'w')
        json.dump(camera_intr_dict, f)
        f.close()
        
    @staticmethod
    def load(filename):
        """ Load the intrinsics from a file """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() != INTR_EXTENSION:
            raise ValueError('Extension %s not supported for CameraIntrinsics. Must be stored with extension %s' %(file_ext, INTR_EXTENSION))

        f = open(filename, 'r')
        ci = json.load(f)
        f.close()
        return CameraIntrinsics(frame=ci['_frame'],
                                fx=ci['_fx'],
                                fy=ci['_fy'],
                                cx=ci['_cx'],
                                cy=ci['_cy'],
                                skew=ci['_skew'],
                                height=ci['_height'],
                                width=ci['_width'])
                                

        
