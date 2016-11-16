"""
Lean classes to encapculate images
Author: Jeff
"""
from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
import PIL.Image as PImage
import IPython
import os
import scipy.misc as sm
import cv2
import matplotlib.pyplot as plt

import scipy.spatial.distance as ssd

from core import RigidTransform, Point, PointCloud, NormalCloud, PointNormalCloud, Box
import core.constants as constants

class Image(object):
    __metaclass__ = ABCMeta    

    def __init__(self, data, frame='unspecified'):
        if not isinstance(data, np.ndarray):
            raise ValueError('Must initialize image with a numpy ndarray')
        if not isinstance(frame, str) and not isinstance(frame, unicode):
            raise ValueError('Must provide string name of frame of data')
        
        self._check_valid_data(data)
        self._data = self._preprocess_data(data)
        self._frame = frame
        
    @abstractmethod
    def _check_valid_data(self, data):
        """ Checks that the data is valid for the class type """
        pass

    @abstractmethod
    def _image_data(self):
        """ Returns the data in image format (scaling and conversion to uint8) """
        pass

    @abstractmethod
    def resize(self, size, interp):
        """ Resize the image """
        pass

    def _preprocess_data(self, data):
        """ Converts data array to the preferred 3-dim structure """
        original_type = data.dtype
        if len(data.shape) == 1:
            data = data[:,np.newaxis,np.newaxis]
        elif len(data.shape) == 2:
            data = data[:,:,np.newaxis]
        elif len(data.shape) == 0 or len(data.shape) > 3:
            raise ValueError('Illegal data array passed to image. Must be 1, 2, or 3 dimensional numpy array')
        return data.astype(original_type)

    @property
    def shape(self):
        return self._data.shape

    @property
    def height(self):
        return self._data.shape[0]

    @property
    def width(self):
        return self._data.shape[1]

    @property
    def channels(self):
        return self._data.shape[2]

    @property
    def type(self):
        return self._data.dtype.type

    @property
    def raw_data(self):
        return self._data

    @property
    def data(self):
        return self._data.squeeze()

    @property
    def frame(self):
        return self._frame

    def ij_to_linear(self, i, j):
        return i + j.dot(self.width)

    def linear_to_ij(self, linear_inds):
        return np.c_[linear_inds / self.width, linear_inds % self.width]

    def mask_by_ind(self, inds):
        new_data = np.zeros(self.shape)
        for ind in inds:
            new_data[ind[0], ind[1]] = self.data[ind[0], ind[1]]
        return type(self)(new_data, self.frame)

    def mask_by_linear_ind(self, linear_inds):
        inds = self.linear_to_ij(linear_inds)
        return self.mask_by_ind(inds)

    def is_same_shape(self, other_im, check_channels=False):
        """ Checks if two images have the same height and width (and optionally channels).
        
        Parameters
        ----------
        other_im : :obj:`Image`
            image to compare
        check_channels : bool
            whether or not to check equality of the channels

        Returns
        -------
        bool
            True if the images are the same shape, False otherwise
        """
        if self.height == other_im.height and self.width == other_im.width:
            if check_channels and self.channels != other_im.channels:
                return False
            return True
        return False                

    @staticmethod
    def median_images(images):
        '''Find the median image'''
        images_data = [image.data for image in images]
        median_image_data = np.median(images_data, axis=0)

        an_image = images[0]
        return type(an_image)(median_image_data.astype(an_image.data.dtype), an_image.frame)

    def __getitem__(self, indices):
        # read indices
        j = None
        k = None
        if type(indices) in (tuple, np.ndarray):
            i = indices[0]
            if len(indices) > 1:
                j = indices[1]
            if len(indices) > 2:
                k = indices[2]
        else:
            i = indices

        # check indices
        if (type(i) == int and i < 0) or \
           (j is not None and type(j) == int and j < 0) or \
           (k is not None and type(k) is int and k < 0) or \
           (type(i) == int and i >= self.height) or \
           (j is not None and type(j) == int and j >= self.width) or \
           (k is not None and type(k) == int and k >= self.channels):
            raise ValueError('Out of bounds indexing')
        if k is not None and type(k) == int and k > 1 and self.channels < 3:
            raise ValueError('Illegal indexing. Image is not 3 dimensional')

        # linear indexing
        if j is None:
            return self._data[i]        
        # return the channel vals for the i, j pixel
        if k is None:
            return self._data[i,j,:]
        return self._data[i,j,k]

    def apply(self, method, *args, **kwargs):
        data = method(self.data, *args, **kwargs)
        return type(self)(data.astype(self.type), self.frame)

    def crop(self, height, width, center_i=None, center_j=None):
        """ Crop the image centered around center_i, center_j """
        if center_i is None:
            center_i = self.height / 2
        if center_j is None:
            center_j = self.width / 2

        start_row = max(0, center_i - height / 2)
        end_row = min(self.height -1, center_i + height / 2)
        start_col = max(0, center_j - width / 2)
        end_col = min(self.width - 1, center_j + width / 2)
        
        return type(self)(self._data[start_row:end_row+1, start_col:end_col+1], self._frame)

    def focus(self, height, width, center_i=None, center_j=None):
        """ Blacks all of the image except within the box """
        if center_i is None:
            center_i = self.height / 2
        if center_j is None:
            center_j = self.width / 2

        start_row = max(0, center_i - height / 2)
        end_row = min(self.height -1, center_i + height / 2)
        start_col = max(0, center_j - width / 2)
        end_col = min(self.width - 1, center_j + width / 2)
        
        focus_data = np.zeros(self._data.shape)
        focus_data[start_row:end_row+1, start_col:end_col+1] = self._data[start_row:end_row+1,
                                                                          start_col:end_col+1]
        return type(self)(focus_data.astype(self._data.dtype), self._frame)

    def center_nonzero(self):
        """ Recenters the image on mean area of nonzero pixels """
        # get the center of the nonzero pixels
        nonzero_px = np.where(self._data != 0.0)
        nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
        mean_px = np.mean(nonzero_px, axis=0)
        center_px = (np.array(self.shape) / 2.0) [:2]
        diff_px = center_px - mean_px

        # transform image
        nonzero_px_tf = nonzero_px + diff_px
        nonzero_px_tf[:,0] = np.max(np.c_[np.zeros(nonzero_px_tf[:,0].shape), nonzero_px_tf[:,0]], axis=1)
        nonzero_px_tf[:,0] = np.min(np.c_[(self.height-1)*np.ones(nonzero_px_tf[:,0].shape), nonzero_px_tf[:,0]], axis=1)
        nonzero_px_tf[:,1] = np.max(np.c_[np.zeros(nonzero_px_tf[:,1].shape), nonzero_px_tf[:,1]], axis=1)
        nonzero_px_tf[:,1] = np.min(np.c_[(self.width-1)*np.ones(nonzero_px_tf[:,1].shape), nonzero_px_tf[:,1]], axis=1)
        nonzero_px = nonzero_px.astype(np.uint16)
        nonzero_px_tf = nonzero_px_tf.astype(np.uint16)
        shifted_data = np.zeros(self.shape)
        shifted_data[nonzero_px_tf[:,0], nonzero_px_tf[:,1], :] = self.data[nonzero_px[:,0], nonzero_px[:,1]].reshape(-1, self.channels)

        return type(self)(shifted_data.astype(self.data.dtype), frame=self._frame), diff_px
        
    def save(self, filename):
        """ Writes the image to file """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext in constants.SUPPORTED_IMAGE_EXTS:
            im_data = self._image_data()
            pil_image = PImage.fromarray(im_data.squeeze())
            pil_image.save(filename)
        elif file_ext == '.npy':
            np.save(filename, self._data)
        elif file_ext == '.npz':
            np.savez_compressed(filename, self._data)
        else:
            raise ValueError('Extension %s not supported' %(file_ext))

    def savefig(self, output_path, title, dpi=400, format='png', cmap=None):
        plt.figure()
        plt.imshow(self.data, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        title_underscore = title.replace(' ', '_')
        plt.savefig(os.path.join(output_path,'{0}.{1}'.format(title_underscore, format)), dpi=dpi, format=format)

    @staticmethod
    def load_data(filename):
        """ Writes the depth image to file """
        file_root, file_ext = os.path.splitext(filename)
        data = None
        if file_ext.lower() in constants.SUPPORTED_IMAGE_EXTS:
            pil_image = PImage.open(filename)
            data = np.array(pil_image)
        elif file_ext == '.npy':
            data = np.load(filename)
        elif file_ext == '.npz':
            data = np.load(filename)['arr_0']
        else:
            raise ValueError('Extension %s not supported' %(file_ext))
        return data

class ColorImage(Image):
    def __init__(self, data, frame='unspecified'):
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """ Checks for float, single channel """
        if data.dtype.type is not np.uint8:
            raise ValueError('Illegal data type. Color images only support uint8 arrays')

        if len(data.shape) == 3 and data.shape[2] != 1 and data.shape[2] != 3:
            raise ValueError('Illegal data type. Color images only support one or three channels')

    def _image_data(self):
        return self._data

    @property
    def r_data(self):
        return self.data[:,:,0]

    @property
    def g_data(self):
        return self.data[:,:,1]

    @property
    def b_data(self):
        return self.data[:,:,2]

    def resize(self, size, interp='bilinear'):
        resized_data = sm.imresize(self.data, size, interp=interp)
        return ColorImage(resized_data, self._frame)

    def find_chessboard(self, sx=6, sy=9):
        """ Finds the corners of an sx X sy chessboard in the image """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((sx*sy,3), np.float32)
        objp[:,:2] = np.mgrid[0:sx,0:sy].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # create images
        img = self.data.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (sx,sy), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            if corners is not None:
                return corners.squeeze()
        return None

    def mask_binary(self, binary_im):
        """ Sets all values where binary_im is zero to 0.0 """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1], :] = 0.0
        return ColorImage(data, self._frame)

    def foreground_mask(self, tolerance, ignore_endpoints=True, use_hsv=False, scale=8, bgmodel=None):
        """
        Creates a binary image mask for the foreground of an image
        against a uniformly colored background.
        The background is assumed to be the mode value of the histogram
        for each of the color channels.
        """
        # get a background model
        if bgmodel is None:
            bgmodel = self.background_model(ignore_endpoints=ignore_endpoints,
                                            use_hsv=use_hsv,
                                            scale=scale)

        # get the bounds
        lower_bound = np.array([bgmodel[i] - tolerance for i in range(self.channels)])
        upper_bound = np.array([bgmodel[i] + tolerance for i in range(self.channels)])
        orig_zero_indices = np.where(self._data == 0)

        # threshold
        binary_data = cv2.inRange(self.data, lower_bound, upper_bound)
        binary_data[:,:,] = (255 - binary_data[:,:,])
        binary_data[orig_zero_indices[0], orig_zero_indices[1],] = 0.0
        binary_im = BinaryImage(binary_data.astype(np.uint8), frame=self.frame)
        return binary_im

    def background_model(self, ignore_endpoints=True, use_hsv=False, scale=8):
        """ Returns a background model based on the image mode """
        # hsv color
        data = self.data
        if use_hsv:
            pil_im = PImage.fromarray(self._data)
            pil_im = pil_im.convert('HSV')
            data = np.asarray(pil_im)
 
        # generate histograms for each channel
        bounds = (0, np.iinfo(np.uint8).max + 1)
        num_bins = bounds[1] / scale
        r_hist, _ = np.histogram(self.r_data, bins=num_bins, range=bounds)
        g_hist, _ = np.histogram(self.g_data, bins=num_bins, range=bounds)
        b_hist, _ = np.histogram(self.b_data, bins=num_bins, range=bounds)
        hists = (r_hist, g_hist, b_hist)

        # find the thesholds as the modes of the image
        modes = [0 for i in range(self.channels)]
        for i in range(self.channels):
            if ignore_endpoints:
                modes[i] = scale * (np.argmax(hists[i][1:-1]) + 1)
            else:
                modes[i] = scale * np.argmax(hists[i])                

        return modes

    def draw_box(self, box):
        """ Draw a white box on the image """
        box_data = self._data.copy()
        min_i = box.min_pt[1]
        min_j = box.min_pt[0]
        max_i = box.max_pt[1]
        max_j = box.max_pt[0]

        #draw the vertical lines
        for j in range(min_j, max_j):
            box_data[min_i,j,:] = 255 * np.ones(self.channels)
            box_data[max_i,j,:] = 255 * np.ones(self.channels)

        #draw the horizontal lines
        for i in range(min_i, max_i):
            box_data[i,min_j,:] = 255 * np.ones(self.channels)
            box_data[i,max_j,:] = 255 * np.ones(self.channels)

        return ColorImage(box_data, self._frame)

    def color_histogram(self, hist_size=np.iinfo(np.uint8).max + 1, ignore_black=True):
        """ Return a color histogram """
        # generate histograms for each channel
        split = cv2.split(self.data)
        hists = np.array([cv2.calcHist([split[i]], [0], None, [hist_size], [0, hist_size]) for i in range(self.channels)])

        modes = [0 for i in range(self.channels)]
        for i in range(self.channels):
            if ignore_black:
                modes[i] = np.argmax(hists[i][1:])
            else:
                modes[i] = np.argmax(hists[i])                

        return modes

    def to_grayscale(self):
        """ Converts the color image to grayscale using OpenCV.

        Returns
        -------
        :obj:`GrayscaleImage`
            grayscale image corresponding to original color image
        """
        gray_data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
        return GrayscaleImage(gray_data, frame=self.frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens the color image """
        data = Image.load_data(filename).astype(np.uint8)
        return ColorImage(data, frame)

class DepthImage(Image):
    def __init__(self, data, frame='unspecified'):
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """ Checks for float, single channel """
        if data.dtype.type is not np.float32 and \
                data.dtype.type is not np.float64:
            raise ValueError('Illegal data type. Depth images only support float arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. Depth images only support single channel')

    def _image_data(self):
        depth_data = (self._data * (255.0 / constants.MAX_DEPTH)).squeeze()
        im_data = np.zeros([self.height, self.width, 3])
        im_data[:,:,0] = depth_data
        im_data[:,:,1] = depth_data
        im_data[:,:,2] = depth_data

        zero_indices = np.where(im_data == 0)
        im_data[zero_indices[0], zero_indices[1]] = 255.0
        return im_data.astype(np.uint8)

    def to_color(self):
        im_data = self._image_data()
        return ColorImage(im_data, frame=self._frame)

    def resize(self, size, interp='bilinear'):
        """ Resize the image by some scale factor """
        resized_data = sm.imresize(self.data, size, interp=interp, mode='F')
        return DepthImage(resized_data, self._frame)

    def gradients(self):
        """ Return the gradient as numpy arrays """
        gx, gy = np.gradient(self.data)
        return gx, gy

    def threshold(self, front_thresh=0.0, rear_thresh=100.0):
        """ Sets all values less than front_thresh and greater than rear_thresh to 0"""
        data = np.copy(self._data)
        data[data < front_thresh] = 0.0
        data[data > rear_thresh] = 0.0
        return DepthImage(data, self._frame)

    def threshold_gradients(self, grad_thresh):
        """ Threshold the image by gradients """
        data = np.copy(self._data)
        gx, gy = self.gradients()
        gradients = np.zeros([gx.shape[0], gx.shape[1], 2])
        gradients[:,:,0] = gx
        gradients[:,:,1] = gy
        gradient_mags = np.linalg.norm(gradients, axis=2)
        ind = np.where(gradient_mags > grad_thresh)
        data[ind[0], ind[1]] = 0.0
        return DepthImage(data, self._frame)        

    def mask_binary(self, binary_im):
        """ Sets all values where binary_im is zero to 0.0 """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1]] = 0.0
        return DepthImage(data, self._frame)

    def to_binary(self, threshold=0.0):
        """ Convert to a binary image """
        data = 255 * (self._data > threshold)
        return BinaryImage(data.astype(np.uint8), self._frame)

    def px_shift_to_rigid_transform(self, camera_intr, diff_px, to_frame):
        """ Convert pixel shift to rigid transform """
        nonzero_source_depth_px = np.where(self._data > 0.0)
        if nonzero_source_depth_px[0].shape[0] == 0:
            return RigidTransform(from_frame=from_frame,
                                  to_frame=self._frame)

        # find closest px to the target 
        nonzero_source_depth_px = np.c_[nonzero_source_depth_px[0], nonzero_source_depth_px[1]]
        center_px = (np.array(self.shape) / 2.0) [:2]
        source_px = center_px - diff_px
        nonzero_px_diffs = nonzero_source_depth_px - source_px
        nonzero_px_diff_norms = np.linalg.norm(nonzero_px_diffs, axis=1)
        ind = np.where(nonzero_px_diff_norms == np.min(nonzero_px_diff_norms))[0][0]
        source_px = nonzero_source_depth_px[ind,:]
        
        # compute corresponding target px
        target_px = source_px + diff_px
        source_depth = self[source_px[0], source_px[1]]
        source_pt = camera_intr.deproject_pixel(source_depth,
                                                Point(np.array([source_px[1], source_px[0]]), self._frame))
        target_pt = camera_intr.deproject_pixel(source_depth,
                                                Point(np.array([target_px[1], target_px[0]]), self._frame))

        # compute rigid transform
        t_im_im_shifted = target_pt.data - source_pt.data
        t_im_im_shifted[2] = 0.0
        T_im_im_shifted = RigidTransform(translation=t_im_im_shifted,
                                         from_frame=self._frame,
                                         to_frame=to_frame)

        return T_im_im_shifted

    def point_normal_cloud(self, camera_intr):
        """ Computes a point cloud and normals cloud from the depth image """
        point_cloud_im = camera_intr.deproject_to_image(self)
        normal_cloud_im = point_cloud_im.normal_cloud_im()
        point_cloud = point_cloud_im.to_point_cloud()
        normal_cloud = normal_cloud_im.to_normal_cloud()
        return PointNormalCloud(point_cloud.data, normal_cloud.data, frame=self._frame)        

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens a depth image """
        file_root, file_ext = os.path.splitext(filename)
        data = Image.load_data(filename)
        if file_ext.lower() in constants.SUPPORTED_IMAGE_EXTS:
            data = (data * (constants.MAX_DEPTH / 255.0)).astype(np.float32)
        return DepthImage(data, frame)

class IrImage(Image):
    def __init__(self, data, frame='unspecified'):
        Image.__init__(self, data, frame)
        
    def _check_valid_data(self, data):
        """ Checks for uint16, single channel """
        if data.dtype.type is not np.uint16:
            raise ValueError('Illegal data type. IR images only support 16-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. IR images only support single channel ')
    
    def _image_data(self):
        return (self._data * (255.0 / constants.MAX_IR)).astype(np.uint8)

    def resize(self, size, interp='bilinear'):
        resized_data = sm.imresize(self._data, size, interp=interp)
        return IrImage(resized_data, self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens an IR image """
        data = Image.load_data(filename)
        data = (data * (constants.MAX_IR / 255.0)).astype(np.uint16)
        return IrImage(data, frame)

class GrayscaleImage(Image):
    def __init__(self, data, frame):
        Image.__init__(self, data, frame)
        
    def _check_valid_data(self, data):
        """ Checks for uint8, single channel """
        if data.dtype.type is not np.uint8:
            raise ValueError('Illegal data type. Grayscale images only support 8-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. Grayscale images only support single channel ')
    
    def _image_data(self):
        return self._data

    def resize(self, size, interp='bilinear'):
        resized_data = sm.imresize(self._data, size, interp=interp)
        return GrayscaleImage(resized_data, self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens a grayscale image """
        data = Image.load_data(filename)
        return GrayscaleImage(data, frame)

class BinaryImage(Image):
    def __init__(self, data, frame='unspecified', threshold=128):
        self._threshold = threshold
        data = 255 * (data > threshold).astype(data.dtype) # binarize
        Image.__init__(self, data, frame)
        
    def _check_valid_data(self, data):
        """ Checks for uint8, single channel """
        if data.dtype.type is not np.uint8:
            raise ValueError('Illegal data type. Binary images only support 8-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. Binary images only support single channel ')
    
    def _image_data(self):
        return self._data.squeeze()

    def resize(self, size, interp='bilinear'):
        resized_data = sm.imresize(self._data, size, interp=interp)
        return BinaryImage(resized_data, self._frame)

    def prune_contours(self, area_thresh=1000.0, dist_thresh=20):
        """ Prunes all binary image connected components with area less than area_thresh """
        # get all contours (connected components) from the binary image
        contours = cv2.findContours(self.data.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours[0])
        middle_pixel = np.array(self.shape)[:2] / 2
        middle_pixel = middle_pixel.reshape(1,2)
        center_contour = None
        pruned_contours = []

        # find which contours need to be pruned
        for i in range(num_contours):
            area = cv2.contourArea(contours[0][i])
            if area > area_thresh:
                # check close to origin
                fill = np.zeros([self.height, self.width, 3])
                cv2.fillPoly(fill, pts=[contours[0][i]], color=(255,255,255))
                nonzero_px = np.where(fill > 0)
                nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
                dists = ssd.cdist(middle_pixel, nonzero_px)
                min_dist = np.min(dists)
                pruned_contours.append((contours[0][i], min_dist))

        if len(pruned_contours) == 0:
            return None

        pruned_contours.sort(key = lambda x: x[1])

        # keep all contours within some distance of the top
        num_contours = len(pruned_contours)
        keep_indices = [0]
        source_coords = pruned_contours[0][0].squeeze().astype(np.float32)
        for i in range(1, num_contours):
            target_coords = pruned_contours[i][0].squeeze().astype(np.float32)
            dists = ssd.cdist(source_coords, target_coords)
            min_dist = np.min(dists)
            if min_dist < dist_thresh:
                keep_indices.append(i)

        # keep the top num_areas pruned contours
        keep_indices = np.unique(keep_indices)
        pruned_contours = [pruned_contours[i][0] for i in keep_indices]

        # mask out bad areas in the image
        pruned_data = np.zeros([self.height, self.width, 3])
        for contour in pruned_contours:
            cv2.fillPoly(pruned_data, pts=[contour], color=(255,255,255))
        pruned_data = pruned_data[:,:,0] # convert back to one channel

        # preserve topology of original image
        orig_zeros = np.where(self.data == 0)
        pruned_data[orig_zeros[0], orig_zeros[1]] = 0
        return BinaryImage(pruned_data.astype(np.uint8), self._frame)

    def find_contours(self, area_thresh=1000.0):
        """ Returns a list of contours with area above the thresh """
        # get all contours (connected components) from the binary image
        contours = cv2.findContours(self.data.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours[0])
        kept_contours = []

        # find which contours need to be pruned
        for i in range(num_contours):
            area = cv2.contourArea(contours[0][i])
            if area > area_thresh:
                px = contours[0][i].squeeze()
                bounding_box = Box(np.min(px, axis=0), np.max(px, axis=0), self._frame)
                kept_contours.append((area, px, bounding_box))
        return kept_contours

    def closest_nonzero_pixel(self, pixel, direction, w=13, t=0.5):
        """ Finds the closest non-zero pixel along the given direction with clearance """
        # create circular structure for checking clearance
        y, x = np.meshgrid(np.arange(w) - w/2, np.arange(w) - w/2)

        cur_px_y = np.ravel(y + pixel[0]).astype(np.uint16)
        cur_px_x = np.ravel(x + pixel[1]).astype(np.uint16)
        occupied = True
        if np.any(cur_px_y >= 0) and np.any(cur_px_y < self.height) and np.any(cur_px_x >= 0) and np.any(cur_px_x < self.width):
            occupied = np.any(self[cur_px_y, cur_px_x] >= self._threshold)
        while occupied:
            pixel = pixel + t * direction
            cur_px_y = np.ravel(y + pixel[0]).astype(np.uint16)
            cur_px_x = np.ravel(x + pixel[1]).astype(np.uint16)
            if np.any(cur_px_y >= 0) and np.any(cur_px_y < self.height) and np.any(cur_px_x >= 0) and np.any(cur_px_x < self.width):
                occupied = np.any(self[cur_px_y, cur_px_x] >= self._threshold)
            else:
                occupied = False

        return pixel

    def to_color(self):
        """ Convert to color """
        color_data = np.zeros([self.height, self.width, 3])
        color_data[:,:,0] = self.data
        color_data[:,:,1] = self.data
        color_data[:,:,2] = self.data
        return ColorImage(color_data.astype(np.uint8), self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens a binary image """
        data = Image.load_data(filename)
        if len(data.shape) > 2 and data.shape[2] > 1:
            data = data[:,:,0]
        return BinaryImage(data, frame)

class PointCloudImage(Image):
    def __init__(self, data, frame='unspecified'):
        Image.__init__(self, data, frame)
        
    def _check_valid_data(self, data):
        """ Checks for float32, single channel """
        if data.dtype.type is not np.float32 and data.dtype.type is not np.float64:
            raise ValueError('Illegal data type. PointCloud images only support 32-bit or 64-bit float arrays')

        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError('Illegal data type. PointCloud images must have three channels')
    
    def _image_data(self):
        raise NotImplementedError('Image conversion not supported for point cloud')

    def resize(self, size, interp='bilinear'):
        resized_data = sm.imresize(self._data, size, interp=interp)
        return PointCloudImage(resized_data, self._frame)

    def to_point_cloud(self):
        return PointCloud(data=self._data.reshape(self.height*self.width, 3).T,
                          frame=self._frame)

    def normal_cloud_im(self):
        """ Generate a normal cloud im from this """
        gx, gy, _ = np.gradient(self.data)
        gx_data = gx.reshape(self.height*self.width, 3)
        gy_data = gy.reshape(self.height*self.width, 3)
        pc_grads = np.cross(gx_data, gy_data) # default to point toward camera
        pc_grad_norms = np.linalg.norm(pc_grads, axis=1)
        normal_data = pc_grads / np.tile(pc_grad_norms[:,np.newaxis], [1, 3])
        normal_im_data = normal_data.reshape(self.height, self.width, 3)
        return NormalCloudImage(normal_im_data, frame=self.frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens a point cloud image """
        data = Image.load_data(filename)
        return PointCloudImage(data, frame)

class NormalCloudImage(Image):
    def __init__(self, data, frame='unspecified'):
        Image.__init__(self, data, frame)
        
    def _check_valid_data(self, data):
        """ Checks for float32, single channel """
        if data.dtype.type is not np.float32 and data.dtype.type is not np.float64:
            raise ValueError('Illegal data type. NormalCloud images only support 32-bit or 64-bit float arrays')

        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError('Illegal data type. NormalCloud images must have three channels')

        if np.any((np.abs(np.linalg.norm(data, axis=2) - 1.0) > 1e-4) & (np.linalg.norm(data, axis=2) != 0.0)):
            raise ValueError('Illegal data. Must have norm=1.0 or norm=0.0')

    def _image_data(self):
        raise NotImplementedError('Image conversion not supported for normal cloud')

    def resize(self, size, interp='bilinear'):
        raise NotImplementedError('Image resizing not supported for normal cloud')

    def to_normal_cloud(self):
        return NormalCloud(data=self._data.reshape(self.height*self.width, 3).T,
                          frame=self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens a point cloud image """
        data = Image.load_data(filename)
        return NormalCloudImage(data, frame)
