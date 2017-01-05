"""
Lean classes to encapculate images
Author: Jeff
"""
from abc import ABCMeta, abstractmethod
import os
import IPython

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as PImage

import scipy.misc as sm
import scipy.signal as ssg
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.spatial.distance as ssd

import sklearn.cluster as sc
import sklearn.mixture as smx

from core import PointCloud, NormalCloud, PointNormalCloud, Box, Contour

import constants as constants

class Image(object):
    """Abstract wrapper class for images.
    """
    __metaclass__ = ABCMeta    

    def __init__(self, data, frame='unspecified'):
        """Create an image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (i.e. R,G,B values). Alternatively, 
            if the matrix is one dimensional, it will be interpreted as an
            N by 1 image with single element list at each pixel,
            and if the matrix is two dimensional, it
            will be a N by M matrix with a single element list at each pixel.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('Must initialize image with a numpy ndarray')
        if not isinstance(frame, str) and not isinstance(frame, unicode):
            raise ValueError('Must provide string name of frame of data')

        self._check_valid_data(data)
        self._data = self._preprocess_data(data)
        self._frame = frame

    def _preprocess_data(self, data):
        """Converts a data array to the preferred 3D structure.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to process.

        Returns
        -------
        :obj:`numpy.ndarray`
            The data re-formatted (if needed) as a 3D matrix

        Raises
        ------
        ValueError
            If the data is not 1, 2, or 3D to begin with.
        """
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
        """:obj:`tuple` of int : The shape of the data array.
        """
        return self._data.shape

    @property
    def height(self):
        """int : The number of rows in the image.
        """
        return self._data.shape[0]

    @property
    def width(self):
        """int : The number of columns in the image.
        """
        return self._data.shape[1]

    @property
    def center(self):
        """:obj:`numpy.ndarray` of int : The xy indices of the center of the
        image.
        """
        return np.array([self.height/2, self.width/2])

    @property
    def channels(self):
        """int : The number of channels in each pixel. For example, RGB images
        have 3 channels.
        """
        return self._data.shape[2]

    @property
    def type(self):
        """:obj:`numpy.dtype` : The data type of the image's elements.
        """
        return self._data.dtype.type

    @property
    def raw_data(self):
        """:obj:`numpy.ndarray` : The 3D array of data. The first dim is rows,
        the second is columns, and the third is pixel channels.
        """
        return self._data

    @property
    def data(self):
        """:obj:`numpy.ndarray` : The data array, but squeezed to get rid of
        extraneous dimensions.
        """
        return self._data.squeeze()

    @property
    def frame(self):
        """:obj:`str` : The frame of reference in which the image resides.
        """
        return self._frame
        """Create a new image by zeroing out data at locations not in the
        given indices.

        Parameters
        ----------
        inds : :obj:`numpy.ndarray` of int
            A 2D ndarray whose first entry is the list of row indices
            and whose second entry is the list of column indices.
            The data at these indices will not be set to zero.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, with data not indexed by inds set
            to zero.
        """

    @abstractmethod
    def _check_valid_data(self, data):
        """Checks that the given data is valid.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.
        """
        pass

    @abstractmethod
    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is the R/G/B entry.
        """
        pass

    @abstractmethod
    def resize(self, size, interp):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')
        """
        pass

    def transform(self, translation, theta):
        """Create a new image by translating and rotating the current image.

        Parameters
        ----------
        translation : :obj:`numpy.ndarray` of float
            The XY translation vector.

        theta : float
            Rotation angle in radians, with positive meaning counter-clockwise.

        Returns
        -------
        :obj:`Image`
            An image of the same type that has been rotated and translated.
        """
        theta = np.rad2deg(theta)
        trans_map = np.float32([[1,0,translation[1]], [0,1,translation[0]]])
        rot_map = cv2.getRotationMatrix2D(tuple(self.center), theta, 1)
        trans_map_aff = np.r_[trans_map, [[0,0,1]]]
        rot_map_aff = np.r_[rot_map, [[0,0,1]]]
        full_map = rot_map_aff.dot(trans_map_aff)
        full_map = full_map[:2,:]
        im_data_tf = cv2.warpAffine(self.data, full_map, (self.height, self.width), flags=cv2.INTER_NEAREST)
        return type(self)(im_data_tf.astype(self.data.dtype), frame=self._frame)

    def ij_to_linear(self, i, j):
        """Converts row / column coordinates to linear indices.

        Parameters
        ----------
        i : :obj:`numpy.ndarray` of int
            A list of row coordinates.

        j : :obj:`numpy.ndarray` of int
            A list of column coordinates.

        Returns
        -------
        :obj:`numpy.ndarray` of int
            A list of linear coordinates.
        """
        return i + j.dot(self.width)

    def linear_to_ij(self, linear_inds):
        """Converts linear indices to row and column coordinates.

        Parameters
        ----------
        linear_inds : :obj:`numpy.ndarray` of int
            A list of linear coordinates.

        Returns
        -------
        :obj:`numpy.ndarray` of int
            A 2D ndarray whose first entry is the list of row indices
            and whose second entry is the list of column indices.
        """
        return np.c_[linear_inds / self.width, linear_inds % self.width]

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

    def mask_by_ind(self, inds):
        """Create a new image by zeroing out data at locations not in the
        given indices.

        Parameters
        ----------
        inds : :obj:`numpy.ndarray` of int
            A 2D ndarray whose first entry is the list of row indices
            and whose second entry is the list of column indices.
            The data at these indices will not be set to zero.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, with data not indexed by inds set
            to zero.
        """
        new_data = np.zeros(self.shape)
        for ind in inds:
            new_data[ind[0], ind[1]] = self.data[ind[0], ind[1]]
        return type(self)(new_data.astype(self.data.dtype), self.frame)

    def mask_by_linear_ind(self, linear_inds):
        """Create a new image by zeroing out data at locations not in the
        given indices.

        Parameters
        ----------
        linear_inds : :obj:`numpy.ndarray` of int
            A list of linear coordinates.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, with data not indexed by inds set
            to zero.
        """
        inds = self.linear_to_ij(linear_inds)
        return self.mask_by_ind(inds)

    def is_same_shape(self, other_im, check_channels=False):
        """Checks if two images have the same height and width
        (and optionally channels).

        Parameters
        ----------
        other_im : :obj:`Image`
            The image to compare against this one.
        check_channels : bool
            Whether or not to check equality of the channels.

        Returns
        -------
        bool
            True if the images are the same shape, False otherwise.
        """
        if self.height == other_im.height and self.width == other_im.width:
            if check_channels and self.channels != other_im.channels:
                return False
            return True
        return False                

    @staticmethod
    def median_images(images):
        """Create a median Image from a list of Images.

        Parameters
        ----------
        :obj:`list` of :obj:`Image`
            A list of Image objects.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type whose data is the median of all of
            the images' data.
        """
        images_data = [image.data for image in images]
        median_image_data = np.median(images_data, axis=0)

        an_image = images[0]
        return type(an_image)(median_image_data.astype(an_image.data.dtype), an_image.frame)

    def __getitem__(self, indices):
        """Index the image's data array.

        Parameters
        ----------
        indices : int or :obj:`tuple` of int
            * int - A linear index.
            * tuple - An ordered index in row, column, and (optionally) channel order.

        Returns
        -------
        item
            The indexed item.

        Raises
        ------
        ValueError
            If the index is poorly formatted or out of bounds.
        """
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
        """Create a new image by applying a function to this image's data.

        Parameters
        ----------
        method : :obj:`function`
            A function to call on the data. This takes in a ndarray
            as its first argument and optionally takes other arguments.
            It should return a modified data ndarray.

        args : arguments
            Additional args for method.

        kwargs : keyword arguments
            Additional keyword arguments for method.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type with new data generated by calling
            method on the current image's data.
        """
        data = method(self.data, *args, **kwargs)
        return type(self)(data.astype(self.type), self.frame)

    def copy(self):
        """ Returns a copy of this image.
        
        Returns
        -------
        :obj:`Image`
            copy of this image
        """
        return type(self)(self.data.copy(), self.frame)

    def crop(self, height, width, center_i=None, center_j=None):
        """Crop the image centered around center_i, center_j.

        Parameters
        ----------
        height : int
            The height of the desired image.

        width : int
            The width of the desired image.

        center_i : int
            The center height point at which to crop. If not specified, the center
            of the image is used.

        center_j : int
            The center width point at which to crop. If not specified, the center
            of the image is used.

        Returns
        -------
        :obj:`Image`
            A cropped Image of the same type.
        """
        # compute crop center px
        height = int(np.round(height))
        width = int(np.round(width))
        if center_i is None:
            center_i = float(self.height) / 2
        if center_j is None:
            center_j = float(self.width) / 2

        # crop using PIL
        desired_start_row = np.floor(center_i - float(height) / 2)
        desired_end_row = np.floor(center_i + float(height) / 2)
        desired_start_col = np.floor(center_j - float(width) / 2)
        desired_end_col = np.floor(center_j + float(width) / 2)

        pil_im = PImage.fromarray(self.data)
        cropped_pil_im = pil_im.crop((desired_start_col,
                                      desired_start_row,
                                      desired_end_col,
                                      desired_end_row))
        crop_data = np.array(cropped_pil_im)

        if crop_data.shape[0] != height or crop_data.shape[1] != width:
            raise ValueError('Crop dims are incorrect')

        return type(self)(crop_data.astype(self.data.dtype), self._frame)

    def focus(self, height, width, center_i=None, center_j=None):
        """Zero out all of the image outside of a crop box.

        Parameters
        ----------
        height : int
            The height of the desired crop box.

        width : int
            The width of the desired crop box.

        center_i : int
            The center height point of the crop box. If not specified, the center
            of the image is used.

        center_j : int
            The center width point of the crop box. If not specified, the center
            of the image is used.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type and size that is zeroed out except
            within the crop box.
        """
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
        """Recenters the image on the mean of the coordinates of nonzero pixels.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type and size that is re-centered
            at the mean location of the non-zero pixels.
        """
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

    def transform(self, translation, theta):
        """ Translate and rotate image.

        Parameters
        ----------
        translation : :obj:`numpy.ndarray`
            2-vector of the translation to perform, in pixels
        theta : float
            amount to rotate the image
        """
        theta = np.rad2deg(theta)
        trans_map = np.float32([[1,0,translation[1]], [0,1,translation[0]]])
        rot_map = cv2.getRotationMatrix2D(tuple(self.center), theta, 1)
        trans_map_aff = np.r_[trans_map, [[0,0,1]]]
        rot_map_aff = np.r_[rot_map, [[0,0,1]]]
        full_map = rot_map_aff.dot(trans_map_aff)
        full_map = full_map[:2,:]
        im_data_tf = cv2.warpAffine(self.data, full_map, (self.height, self.width), flags=cv2.INTER_NEAREST)
        return type(self)(im_data_tf.astype(self.data.dtype), frame=self._frame)

    def nonzero_pixels(self):
        """ Return an array of the nonzero pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the nonzero pixels
        """
        nonzero_px = np.where(self.data > 0)
        nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
        return nonzero_px

    def zero_pixels(self):
        """ Return an array of the zero pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the zero pixels
        """
        zero_px = np.where(self.data == 0)
        zero_px = np.c_[zero_px[0], zero_px[1]]
        return zero_px

    def finite_pixels(self):
        """ Return an array of the finite pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the finite pixels
        """
        finite_px = np.where(np.isfinite(self.data))
        finite_px = np.c_[finite_px[0], finite_px[1]]
        return finite_px

    def nonzero_data(self):
        """ Returns the values in the image at the nonzero pixels

        Returns
        -------
        :obj:`numpy.ndarray`
             NxC array of the nonzero data
        """
        nonzero_px = self.nonzero_pixels()
        return self.data[nonzero_px[:,0], nonzero_px[:,1], ...]

    def replace_zeros(self, val):
        """ Replaces all zeros in the image with a specified value

        Returns
        -------
        image dtype
             value to replace zeros with
        """
        new_data = self.data.copy()
        new_data[new_data == 0] = val
        return type(self)(new_data.astype(self.data.dtype), frame=self._frame)
        
    def save(self, filename):
        """Writes the image to a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to save the image to. Must be one of .png, .jpg,
            .npy, or .npz.

        Raises
        ------
        ValueError
            If an unsupported file type is specified.
        """
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
        """Write the image to a file using pyplot.

        Parameters
        ----------
        output_path : :obj:`str`
            The directory in which to place the file.

        title : :obj:`str`
            The title of the file in which to save the image.

        dpi : int
            The resolution in dots per inch.

        format : :obj:`str`
            The file format to save. Available options include .png, .pdf, .ps,
            .eps, and .svg.

        cmap : :obj:`Colormap`, optional
            A Colormap object fo the pyplot.
        """
        plt.figure()
        plt.imshow(self.data, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        title_underscore = title.replace(' ', '_')
        plt.savefig(os.path.join(output_path,'{0}.{1}'.format(title_underscore, format)), dpi=dpi, format=format)

    @staticmethod
    def load_data(filename):
        """Loads a data matrix from a given file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        Returns
        -------
        :obj:`numpy.ndarray`
            The data array read from the file.
        """
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
    """An RGB color image.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a color image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (i.e. R,G,B values). Alternatively, the
            image may have a single channel, in which case it is interpreted as
            greyscale.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a uint8 array with one or three
        channels.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.uint8:
            raise ValueError('Illegal data type. Color images only support uint8 arrays')

        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError('Illegal data type. Color images only support three channels')

    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is the R/G/B entry.
        """
        return self._data

    @property
    def r_data(self):
        """:obj:`numpy.ndarray` of uint8 : The red-channel data.
        """
        return self.data[:,:,0]

    @property
    def g_data(self):
        """:obj:`numpy.ndarray` of uint8 : The green-channel data.
        """
        return self.data[:,:,1]

    @property
    def b_data(self):
        """:obj:`numpy.ndarray` of uint8 : The blue-channel data.
        """
        return self.data[:,:,2]

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`ColorImage`
            The resized image.
        """
        resized_data = sm.imresize(self.data, size, interp=interp)
        return ColorImage(resized_data, self._frame)

    def find_chessboard(self, sx=6, sy=9):
        """Finds the corners of an sx X sy chessboard in the image.

        Parameters
        ----------
        sx : int
            Number of chessboard corners in x-direction.
        sy : int
            Number of chessboard corners in y-direction.

        Returns
        -------
        :obj:`list` of :obj:`numpy.ndarray`
            A list containing the 2D points of the corners of the detected
            chessboard, or None if no chessboard found.
        """
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
        """Create a new image by zeroing out data at locations
        where binary_im == 0.0.

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            A BinaryImage of the same size as this image, with pixel values of either
            zero or one. Wherever this image has zero pixels, we'll zero out the
            pixels of the new image.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, masked by the given binary image.
        """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1], :] = 0.0
        return ColorImage(data, self._frame)

    def foreground_mask(self, tolerance, ignore_black=True, use_hsv=False, scale=8, bgmodel=None):
        """Creates a binary image mask for the foreground of an image against
        a uniformly colored background. The background is assumed to be the mode value of the histogram
        for each of the color channels.

        Parameters
        ----------
        tolerance : int
            A +/- level from the detected mean backgroud color. Pixels withing
            this range will be classified as background pixels and masked out.

        ignore_black : bool
            If True, the zero pixels will be ignored
            when computing the background model.

        use_hsv : bool
            If True, image will be converted to HSV for background model
            generation.

        scale : int
            Size of background histogram bins -- there will be 255/size bins
            in the color histogram for each channel.

        bgmodel : :obj:`list` of int
            A list containing the red, green, and blue channel modes of the
            background. If this is None, a background model will be generated
            using the other parameters.

        Returns
        -------
        :obj:`BinaryImage`
            A binary image that masks out the background from the current
            ColorImage.
        """
        # get a background model
        if bgmodel is None:
            bgmodel = self.background_model(ignore_black=ignore_black,
                                            use_hsv=use_hsv,
                                            scale=scale)

        # get the bounds
        lower_bound = np.array([bgmodel[i] - tolerance for i in range(self.channels)])
        upper_bound = np.array([bgmodel[i] + tolerance for i in range(self.channels)])
        orig_zero_indices = np.where(np.sum(self._data, axis=2) == 0)

        # threshold
        binary_data = cv2.inRange(self.data, lower_bound, upper_bound)
        binary_data[:,:,] = (255 - binary_data[:,:,])
        binary_data[orig_zero_indices[0], orig_zero_indices[1],] = 0.0
        binary_im = BinaryImage(binary_data.astype(np.uint8), frame=self.frame)
        return binary_im

    def background_model(self, ignore_black=True, use_hsv=False, scale=8):
        """Creates a background model for the given image. The background
        color is given by the modes of each channel's histogram.

        Parameters
        ----------
        ignore_black : bool
            If True, the zero pixels will be ignored
            when computing the background model.

        use_hsv : bool
            If True, image will be converted to HSV for background model
            generation.

        scale : int
            Size of background histogram bins -- there will be 255/size bins
            in the color histogram for each channel.

        Returns
        -------
            A list containing the red, green, and blue channel modes of the
            background.
        """
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
            if ignore_black:
                modes[i] = scale * (np.argmax(hists[i][1:]) + 1)
            else:
                modes[i] = scale * np.argmax(hists[i])

        return modes

    def draw_box(self, box):
        """Draw a white box on the image.

        Parameters
        ----------
        :obj:`core.Box`
            A 2D box to draw in the image.

        Returns
        -------
        :obj:`ColorImage`
            A new image that is the same as the current one, but with
            the white box drawn in.
        """
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

    def nonzero_hsv_data(self):
        """ Computes non zero hsv values.

        Returns
        -------
        :obj:`numpy.ndarray`
            array of the hsv values for the image
        """
        hsv_data = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV)
        nonzero_px = self.nonzero_pixels()
        return hsv_data[nonzero_px[:,0], nonzero_px[:,1], ...]

    def segment_kmeans(self, rgb_weight, num_clusters, hue_weight=0.0):
        """
        Segment a color image using KMeans based on spatial and color distances.
        Black pixels will automatically be assigned to their own 'background' cluster.

        Parameters
        ----------
        rgb_weight : float
            weighting of RGB distance relative to spatial and hue distance
        num_clusters : int
            number of clusters to use
        hue_weight : float
            weighting of hue from hsv relative to spatial and RGB distance

        Returns
        -------
        :obj:`SegmentationImage`
            image containing the segment labels
        """
        # form features array
        label_offset = 1
        nonzero_px = np.where(self.data != 0.0)
        nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]

        # get hsv data if specified
        color_vals = rgb_weight * self._data[nonzero_px[:,0], nonzero_px[:,1], :]
        if hue_weight > 0.0:
            hsv_data = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV)
            color_vals = np.c_[color_vals, hue_weight * hsv_data[nonzero_px[:,0], nonzero_px[:,1], :1]]
        features = np.c_[nonzero_px, color_vals.astype(np.float32)]

        # perform KMeans clustering
        kmeans = sc.KMeans(n_clusters=num_clusters)
        labels = kmeans.fit_predict(features)

        # create output label array
        label_im = np.zeros([self.height, self.width]).astype(np.uint8)
        label_im[nonzero_px[:,0], nonzero_px[:,1]] = labels + label_offset
        return SegmentationImage(label_im, frame=self.frame)

    def inpaint(self, win_size=3, rescale_factor=1.0):
        """ Fills in the zero pixels in the image.

        Parameters
        ----------
        win_size : int
            size of window to use for inpainting
        rescale_factor : float
            amount to rescale the image for inpainting, smaller numbers increase speed

        Returns
        -------
        :obj:`ColorImage`
            color image with zero pixels filled in
        """
        # resize the image
        resized_data = self.resize(rescale_factor, interp='nearest').data

        # inpaint smaller image
        mask = 1 * (np.sum(resized_data, axis=2) == 0)
        inpainted_data = cv2.inpaint(resized_data, mask.astype(np.uint8),
                                     win_size, cv2.INPAINT_TELEA)
        inpainted_im = ColorImage(inpainted_data, frame=self.frame)

        # fill in zero pixels with inpainted and resized image
        filled_data = inpainted_im.resize(1.0 / rescale_factor, interp='bilinear').data
        new_data = self.data
        new_data[self.data == 0] = filled_data[self.data == 0]
        return ColorImage(new_data, frame=self.frame)

    def to_binary(self, threshold=0.0):
        """Converts the color image to binary.

        Returns
        -------
        :obj:`BinaryImage`
            Binary image corresponding to the nonzero px of the original image
        """
        data = 255 * (self._data > threshold)
        return BinaryImage(data[:,:,0].astype(np.uint8), self._frame)

    def to_grayscale(self):
        """Converts the color image to grayscale using OpenCV.

        Returns
        -------
        :obj:`GrayscaleImage`
            Grayscale image corresponding to original color image.
        """
        gray_data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
        return GrayscaleImage(gray_data, frame=self.frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a ColorImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`ColorImage`
            The new color image.
        """
        data = Image.load_data(filename).astype(np.uint8)
        return ColorImage(data, frame)

class DepthImage(Image):
    """A depth image in which individual pixels have a single floating-point
    depth channel.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a depth image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (depths as floating point numbers).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)
        self._data = self._data.astype(np.float32)

    def _check_valid_data(self, data):
        """Checks that the given data is a float array with one channel.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.float32 and \
                data.dtype.type is not np.float64:
            raise ValueError('Illegal data type. Depth images only support float arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. Depth images only support single channel')

    def _image_data(self, normalize=False):
        """Returns the data in image format, with scaling and conversion to uint8 types.
        
        Parameters
        ----------
        normalize : bool
            whether or not to normalize by the min and max depth of the image

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is a set of 3 RGB values, each of
            which is simply the depth entry scaled to between 0 and 255.
        """
        if normalize:
            min_depth = np.min(self._data)
            max_depth = np.max(self._data)
            depth_data = (self._data - min_depth) / (max_depth - min_depth)
            depth_data = 255.0 * depth_data.squeeze()
        else:
            depth_data = (self._data * (255.0 / constants.MAX_DEPTH)).squeeze()
        im_data = np.zeros([self.height, self.width, 3])
        im_data[:,:,0] = depth_data
        im_data[:,:,1] = depth_data
        im_data[:,:,2] = depth_data

        zero_indices = np.where(im_data == 0)
        im_data[zero_indices[0], zero_indices[1]] = 255.0
        return im_data.astype(np.uint8)

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`DepthImage`
            The resized image.
        """
        resized_data = sm.imresize(self.data, size, interp=interp, mode='F')
        return DepthImage(resized_data, self._frame)

    def gradients(self):
        """Return the gradient as a pair of numpy arrays.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float
            The x-gradient and y-gradient of the image.
        """
        gx, gy = np.gradient(self.data)
        return gx, gy

    def threshold(self, front_thresh=0.0, rear_thresh=100.0):
        """Creates a new DepthImage by setting all depths less than
        front_thresh and greater than rear_thresh to 0.

        Parameters
        ----------
        front_thresh : float
            The lower-bound threshold.

        rear_thresh : float
            The upper bound threshold.

        Returns
        -------
        :obj:`DepthImage`
            A new DepthImage created from the thresholding operation.
        """
        data = np.copy(self._data)
        data[data < front_thresh] = 0.0
        data[data > rear_thresh] = 0.0
        return DepthImage(data, self._frame)

    def threshold_gradients(self, grad_thresh):
        """Creates a new DepthImage by zeroing out all depths
        where the magnitude of the gradient at that point is
        greater than grad_thresh.

        Parameters
        ----------
        grad_thresh : float
            A threshold for the gradient magnitude.

        Returns
        -------
        :obj:`DepthImage`
            A new DepthImage created from the thresholding operation.
        """
        data = np.copy(self._data)
        gx, gy = self.gradients()
        gradients = np.zeros([gx.shape[0], gx.shape[1], 2])
        gradients[:,:,0] = gx
        gradients[:,:,1] = gy
        gradient_mags = np.linalg.norm(gradients, axis=2)
        ind = np.where(gradient_mags > grad_thresh)
        data[ind[0], ind[1]] = 0.0
        return DepthImage(data, self._frame)        

    def threshold_gradients_pctile(self, thresh_pctile, min_mag=0.0):
        """Creates a new DepthImage by zeroing out all depths
        where the magnitude of the gradient at that point is
        greater than some percentile of all gradients.

        Parameters
        ----------
        thresh_pctile : float
            percentile to threshold all gradients above
        min_mag : float
            minimum magnitude of the gradient

        Returns
        -------
        :obj:`DepthImage`
            A new DepthImage created from the thresholding operation.
        """
        data = np.copy(self._data)
        gx, gy = self.gradients()
        gradients = np.zeros([gx.shape[0], gx.shape[1], 2])
        gradients[:,:,0] = gx
        gradients[:,:,1] = gy
        gradient_mags = np.linalg.norm(gradients, axis=2)
        grad_thresh = np.percentile(gradient_mags, thresh_pctile)
        ind = np.where((gradient_mags > grad_thresh) & (gradient_mags > min_mag))
        data[ind[0], ind[1]] = 0.0
        return DepthImage(data, self._frame)        

    def inpaint(self, rescale_factor=1.0):
        """ Fills in the zero pixels in the image.

        Parameters
        ----------
        rescale_factor : float
            amount to rescale the image for inpainting, smaller numbers increase speed

        Returns
        -------
        :obj:`DepthImage`
            depth image with zero pixels filled in
        """
        # form inpaint kernel
        inpaint_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])

        # resize the image
        resized_data = self.resize(rescale_factor, interp='nearest').data

        # inpaint the smaller image
        cur_data = resized_data.copy()
        zeros = (cur_data == 0)
        while np.any(zeros):
            neighbors = ssg.convolve2d((cur_data != 0), inpaint_kernel,
                                       mode='same', boundary='symm')
            avg_depth = ssg.convolve2d(cur_data, inpaint_kernel,
                                       mode='same', boundary='symm')
            avg_depth[neighbors>0] = avg_depth[neighbors>0] / neighbors[neighbors>0]
            avg_depth[neighbors==0] = 0
            avg_depth[resized_data > 0] = resized_data[resized_data > 0]
            cur_data = avg_depth
            zeros = (cur_data == 0)

        # fill in zero pixels with inpainted and resized image
        inpainted_im = DepthImage(cur_data, frame=self.frame)
        filled_data = inpainted_im.resize(1.0 / rescale_factor, interp='bilinear').data
        new_data = self.data
        new_data[self.data == 0] = filled_data[self.data == 0]
        return DepthImage(new_data, frame=self.frame)

    def mask_binary(self, binary_im):
        """Create a new image by zeroing out data at locations
        where binary_im == 0.0.

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            A BinaryImage of the same size as this image, with pixel values of either
            zero or one. Wherever this image has zero pixels, we'll zero out the
            pixels of the new image.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, masked by the given binary image.
        """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1]] = 0.0
        return DepthImage(data, self._frame)

    def combine_with(self, depth_im):
        """
        Replaces all zeros in the source depth image with the value of a different depth image

        Parameters
        ----------
        depth_im : :obj:`DepthImage`
            depth image to combine with

        Returns
        -------
        :obj:`DepthImage`
            the combined depth image
        """
        new_data = self.data.copy()
        new_data[new_data == 0] = depth_im.data[new_data == 0]
        return DepthImage(new_data, frame=self.frame)

    def to_binary(self, threshold=0.0):
        """Creates a BinaryImage from the depth image. Points where the depth
        is greater than threshold are converted to ones, and all other points
        are zeros.

        Parameters
        ----------
        threshold : float
            The depth threshold.

        Returns
        -------
        :obj:`BinaryImage`
            A BinaryImage where all 1 points had a depth greater than threshold
            in the DepthImage.
        """
        data = 255 * (self._data > threshold)
        return BinaryImage(data.astype(np.uint8), self._frame)

    def to_color(self, normalize=False):
        """ Convert to a color image.

        Parameters
        ----------
        normalize : bool
             whether or not to normalize by the maximum depth
        
        Returns
        -------
        :obj:`ColorImage`
            color image corresponding to the depth image
        """
        im_data = self._image_data(normalize=normalize)
        return ColorImage(im_data, frame=self._frame)

    def to_float(self):
        """ Converts to 32-bit data.

        Returns
        -------
        :obj:`DepthImage`
            depth image with 32 bit float data
        """
        return DepthImage(self.data.astype(np.float32), frame=self.frame)

    def point_normal_cloud(self, camera_intr):
        """Computes a PointNormalCloud from the depth image.

        Parameters
        ----------
        camera_intr : :obj:`CameraIntrinsics`
            The camera parameters on which this depth image was taken.

        Returns
        -------
        :obj:`core.PointNormalCloud`
            A PointNormalCloud created from the depth image.
        """
        point_cloud_im = camera_intr.deproject_to_image(self)
        normal_cloud_im = point_cloud_im.normal_cloud_im()
        point_cloud = point_cloud_im.to_point_cloud()
        normal_cloud = normal_cloud_im.to_normal_cloud()
        return PointNormalCloud(point_cloud.data, normal_cloud.data, frame=self._frame)        

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a DepthImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`DepthImage`
            The new depth image.
        """
        file_root, file_ext = os.path.splitext(filename)
        data = Image.load_data(filename)
        if file_ext.lower() in constants.SUPPORTED_IMAGE_EXTS:
            data = (data * (constants.MAX_DEPTH / 255.0)).astype(np.float32)
        return DepthImage(data, frame)

class IrImage(Image):
    """An IR image in which individual pixels have a single uint16 channel.
    """

    def __init__(self, data, frame='unspecified'):
        """Create an IR image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (IR values as uint16's).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a uint16 array with one channel.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.uint16:
            raise ValueError('Illegal data type. IR images only support 16-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. IR images only support single channel ')

    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is simply the IR entry scaled to between 0 and 255.
        """
        return (self._data * (255.0 / constants.MAX_IR)).astype(np.uint8)

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`IrImage`
            The resized image.
        """
        resized_data = sm.imresize(self._data, size, interp=interp)
        return IrImage(resized_data, self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates an IrImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`IrImage`
            The new IR image.
        """
        data = Image.load_data(filename)
        data = (data * (constants.MAX_IR / 255.0)).astype(np.uint16)
        return IrImage(data, frame)

class GrayscaleImage(Image):
    """A grayscale image in which individual pixels have a single uint8 channel.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a grayscale image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (greyscale values as uint8's).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a uint8 array with one channel.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.uint8:
            raise ValueError('Illegal data type. Grayscale images only support 8-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. Grayscale images only support single channel ')

    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is simply the greyscale entry
            scaled to between 0 and 255.
        """
        return self._data

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`GrayscaleImage`
            The resized image.
        """
        resized_data = sm.imresize(self._data, size, interp=interp)
        return GrayscaleImage(resized_data, self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a GrayscaleImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`GrayscaleImage`
            The new grayscale image.
        """
        data = Image.load_data(filename)
        return GrayscaleImage(data, frame)

class BinaryImage(Image):
    """A binary image in which individual pixels are either black or white (0 or 255).
    """

    def __init__(self, data, frame='unspecified', threshold=128):
        """Create a BinaryImage image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (only one channel, all uint8).
            The data array will be thresholded
            and will end up only containing elements that are 255 or 0.

        threshold : int
            A threshold value. Any value in the data array greater than
            threshold will be set to 255, and all others will be set to 0.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        self._threshold = threshold
        data = 255 * (data > threshold).astype(data.dtype) # binarize
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a uint8 array with one channel.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.uint8:
            raise ValueError('Illegal data type. Binary images only support 8-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. Binary images only support single channel ')

    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is simply the binary 0/255 value.
        """
        return self._data.squeeze()

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`BinaryImage`
            The resized image.
        """
        resized_data = sm.imresize(self._data, size, interp=interp)
        return BinaryImage(resized_data, self._frame)


    def mask_binary(self, binary_im):
        """ Takes AND operation with other binary image.
        
        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            binary image for and operation
       
        Returns
        -------
        :obj:`BinaryImage`
            AND of this binary image and other image
        """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1], ...] = 0.0
        return BinaryImage(data, self._frame)

    def prune_contours(self, area_thresh=1000.0, dist_thresh=20):
        """Removes all white connected components with area less than area_thresh.

        Parameters
        ----------
        area_thresh : float
            The minimum area for which a white connected component will not be
            zeroed out.

        dist_thresh : int
            If a connected component is within dist_thresh of the top of the
            image, it will not be pruned out, regardless of its area.

        Returns
        -------
        :obj:`BinaryImage`
            The new pruned binary image.
        """
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

    def find_contours(self, min_area=0.0, max_area=np.inf):
        """Returns a list of connected components with an area between
        min_area and max_area.

        Parameters
        ----------
        min_area : float
            The minimum area for a contour
        max_area : float
            The maximum area for a contour

        Returns
        -------
        :obj:`list` of :obj:`Contour`
            A list of resuting contours
        """
        # get all contours (connected components) from the binary image
        contours = cv2.findContours(self.data.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours[0])
        kept_contours = []
        
        # find which contours need to be pruned
        for i in range(num_contours):
            area = cv2.contourArea(contours[0][i])
            if area > min_area and area < max_area:
                boundary_px = contours[0][i].squeeze()
                boundary_px_ij_swapped = np.zeros(boundary_px.shape)
                boundary_px_ij_swapped[:,0] = boundary_px[:,1]
                boundary_px_ij_swapped[:,1] = boundary_px[:,0]
                kept_contours.append(Contour(boundary_px_ij_swapped, area=area, frame=self._frame))        

        return kept_contours

    def closest_nonzero_pixel(self, pixel, direction, w=13, t=0.5):
        """Starting at pixel, moves pixel by direction * t until there is a
        non-zero pixel within a radius w of pixel. Then, returns pixel.

        Parameters
        ----------
        pixel : :obj:`numpy.ndarray` of float
            The initial pixel location at which to start.

        direction : :obj:`numpy.ndarray` of float
            The 2D direction vector in which to move pixel.

        w : int
            A circular radius in which to check for non-zero pixels.
            As soon as the current pixel has some non-zero pixel with a raidus
            w of it, this function returns the current pixel location.

        t : float
            The step size with which to move pixel along direction.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The first pixel location along the direction vector at which there
            exists some non-zero pixel within a radius w.
        """
        # create circular structure for checking clearance
        y, x = np.meshgrid(np.arange(w) - w/2, np.arange(w) - w/2)

        cur_px_y = np.ravel(y + pixel[0]).astype(np.uint16)
        cur_px_x = np.ravel(x + pixel[1]).astype(np.uint16)
        occupied = True
        if np.all(cur_px_y >= 0) and np.all(cur_px_y < self.height) and np.all(cur_px_x >= 0) and np.all(cur_px_x < self.width):
            occupied = np.all(self[cur_px_y, cur_px_x] >= self._threshold)
        while occupied:
            pixel = pixel + t * direction
            cur_px_y = np.ravel(y + pixel[0]).astype(np.uint16)
            cur_px_x = np.ravel(x + pixel[1]).astype(np.uint16)
            if np.all(cur_px_y >= 0) and np.all(cur_px_y < self.height) and np.all(cur_px_x >= 0) and np.all(cur_px_x < self.width):
                occupied = np.all(self[cur_px_y, cur_px_x] >= self._threshold)
            else:
                occupied = False

        return pixel

    def add_frame(self, left_boundary, right_boundary, upper_boundary, lower_boundary):
        """ Adds a frame to the image, e.g. turns the boundaries white
        
        Parameters
        ----------
        left_boundary : int
            the leftmost boundary of the frame
        right_boundary : int
            the rightmost boundary of the frame (must be greater than left_boundary)
        upper_boundary : int
            the upper boundary of the frame
        lower_boundary : int
            the lower boundary of the frame (must be greater than upper_boundary)

        Returns
        -------
        :obj:`BinaryImage`
            binary image with white (255) on the boundaries
        """
        # check valid boundary pixels
        left_boundary = max(0, left_boundary)
        right_boundary = min(self.width-1, right_boundary)
        upper_boundary = max(0, upper_boundary)
        lower_boundary = min(self.height-1, lower_boundary)

        if right_boundary < left_boundary:
            raise ValueError('Left boundary must be smaller than the right boundary')
        if upper_boundary > lower_boundary:
            raise ValueError('Upper boundary must be smaller than the lower boundary')

        # fill in border pixels
        bordered_data = self.data.copy()
        bordered_data[:upper_boundary,:] = 255
        bordered_data[lower_boundary:,:] = 255
        bordered_data[:,:left_boundary] = 255
        bordered_data[:,right_boundary:] = 255
        return BinaryImage(bordered_data, frame=self._frame)

    def most_free_pixel(self):
        """ Find the black pixel with the largest distance from the white pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
            2-vector containing the most free pixel
        """
        dist_tf = snm.distance_transform_edt(255 - self.data)
        max_px = np.where(dist_tf == np.max(dist_tf))
        free_pixel = np.array([max_px[0][0], max_px[1][0]])
        return free_pixel

    def diff_with_target(self, binary_im):
        """ Creates a color image to visualize the overlap between two images.
        Nonzero pixels that match in both images are green.
        Nonzero pixels of this image that aren't in the other image are yellow
        Nonzero pixels of the other image that aren't in this image are red

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            binary image to take the difference with
        
        Returns
        -------
        :obj:`ColorImage`
            color image to visualize the image difference
        """
        red = np.array([255,0,0])
        yellow = np.array([255,255,0])
        green = np.array([0,255,0])
        overlap_data = np.zeros([self.height, self.width, 3])
        unfilled_px = np.where((self.data == 0) & (binary_im.data > 0))
        overlap_data[unfilled_px[0], unfilled_px[1], :] = red
        filled_px = np.where((self.data > 0) & (binary_im.data > 0))
        overlap_data[filled_px[0], filled_px[1], :] = green
        spurious_px = np.where((self.data > 0) & (binary_im.data == 0))
        overlap_data[spurious_px[0], spurious_px[1], :] = yellow        
        return ColorImage(overlap_data.astype(np.uint8), frame=self.frame)

    def to_color(self):
        """Creates a ColorImage from the binary image.

        Returns
        -------
        :obj:`ColorImage`
            The newly-created color image.
        """
        color_data = np.zeros([self.height, self.width, 3])
        color_data[:,:,0] = self.data
        color_data[:,:,1] = self.data
        color_data[:,:,2] = self.data
        return ColorImage(color_data.astype(np.uint8), self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a BinaryImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`BinaryImage`
            The new binary image.
        """
        data = Image.load_data(filename)
        if len(data.shape) > 2 and data.shape[2] > 1:
            data = data[:,:,0]
        return BinaryImage(data, frame)

class SegmentationImage(Image):
    """An image containing integer-valued segment labels.
    """
    def __init__(self, data, frame='unspecified'):
        """Create a BinaryImage image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (only one channel, all uint8).
            The integer-valued data should correspond to segment labels.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        self._num_segments = np.max(data)+1
        Image.__init__(self, data, frame)
        
    def _check_valid_data(self, data):
        """ Checks for uint8, single channel """
        if data.dtype.type is not np.uint8:
            raise ValueError('Illegal data type. Segmentation images only support 8-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Illegal data type. Segmentation images only support single channel ')
    
    @property
    def num_segments(self):
        return self._num_segments

    def _image_data(self):
        return self._data

    def border_pixels(self, grad_sigma=0.5, grad_lower_thresh=0.1, grad_upper_thresh=1.0):
        """
        Returns the pixels on the boundary between all segments, excluding the zero segment.
        
        Parameters
        ----------
        grad_sigma : float
            standard deviation used for gaussian gradient filter
        grad_lower_thresh : float
            lower threshold on gradient threshold used to determine the boundary pixels
        grad_upper_thresh : float
            upper threshold on gradient threshold used to determine the boundary pixels

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of pixels on the boundary
        """
        # boundary pixels
        boundary_im = np.ones(self.shape)
        for i in range(1, self.num_segments):
            label_border_im = self.data.copy()
            label_border_im[self.data == 0] = i
            grad_mag = sf.gaussian_gradient_magnitude(label_border_im.astype(np.float32), sigma=grad_sigma)

            nonborder_px = np.where((grad_mag < grad_lower_thresh) | (grad_mag > grad_upper_thresh))
            boundary_im[nonborder_px[0], nonborder_px[1]] = 0

        # return boundary pixels
        border_px = np.where(boundary_im > 0)
        border_px = np.c_[border_px[0], border_px[1]]
        return border_px

    def segment_mask(self, segnum):
        """ Returns a binary image of just the segment corresponding to the given number.
        
        Parameters
        ----------
        segnum : int
            the number of the segment to generate a mask for
        
        Returns
        -------
        :obj:`BinaryImage`
             binary image data
        """
        binary_data = np.zeros(self.shape)
        binary_data[self.data == segnum+1] = 255
        return BinaryImage(binary_data.astype(np.uint8), frame=self.frame)

    def resize(self, size, interp='nearest'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')
        """
        resized_data = sm.imresize(self.data, size, interp=interp, mode='L')
        return SegmentationImage(resized_data, self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens a segmentation image """
        data = Image.load_data(filename)
        return SegmentationImage(data, frame)

class PointCloudImage(Image):
    """A point cloud image in which individual pixels have three float channels.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a PointCloudImage image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (three floats).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a float array with three channels.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.float32 and data.dtype.type is not np.float64:
            raise ValueError('Illegal data type. PointCloud images only support 32-bit or 64-bit float arrays')

        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError('Illegal data type. PointCloud images must have three channels')

    def _image_data(self):
        """This method is not implemented for PointCloudImages.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError('Image conversion not supported for point cloud')

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`PointCloudImage`
            The resized image.
        """
        resized_data = sm.imresize(self._data, size, interp=interp)
        return PointCloudImage(resized_data, self._frame)

    def to_point_cloud(self):
        """Convert the image to a PointCloud object.

        Returns
        -------
        :obj:`core.PointCloud`
            The corresponding PointCloud.
        """
        return PointCloud(data=self._data.reshape(self.height*self.width, 3).T,
                          frame=self._frame)

    def normal_cloud_im(self):
        """Generate a NormalCloudImage from the PointCloudImage.

        Returns
        -------
        :obj:`NormalCloudImage`
            The corresponding NormalCloudImage.
        """
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
        """Creates a PointCloudImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`PointCloudImage`
            The new PointCloudImage.
        """
        data = Image.load_data(filename)
        return PointCloudImage(data, frame)

class NormalCloudImage(Image):
    """A normal cloud image in which individual pixels have three float channels.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a NormalCloudImage image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (three floats).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a float array with three channels.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.float32 and data.dtype.type is not np.float64:
            raise ValueError('Illegal data type. NormalCloud images only support 32-bit or 64-bit float arrays')

        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError('Illegal data type. NormalCloud images must have three channels')

        if np.any((np.abs(np.linalg.norm(data, axis=2) - 1.0) > 1e-4) & (np.linalg.norm(data, axis=2) != 0.0)):
            raise ValueError('Illegal data. Must have norm=1.0 or norm=0.0')

    def _image_data(self):
        """This method is not implemented for NormalCloudImage.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError('Image conversion not supported for normal cloud')

    def resize(self, size, interp='bilinear'):
        """This method is not implemented for NormalCloudImage.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError('Image resizing not supported for normal cloud')

    def to_normal_cloud(self):
        """Convert the image to a NormalCloud object.

        Returns
        -------
        :obj:`core.NormalCloud`
            The corresponding NormalCloud.
        """
        return NormalCloud(data=self._data.reshape(self.height*self.width, 3).T,
                          frame=self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a NormalCloudImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`NormalCloudImage`
            The new NormalCloudImage.
        """
        data = Image.load_data(filename)
        return NormalCloudImage(data, frame)
