"""
Wrappers for keras classes
Author: Jeff Mahler
"""
import argparse
import copy
import cPickle as pkl
import logging
import IPython
import numpy as np
import os
import random
import sys
import time

import scipy.misc as sm
import scipy.stats as ss

from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Input, GlobalAveragePooling2D, Reshape
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, Iterator, transform_matrix_offset_center, apply_transform
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import SGD
from keras.utils import to_categorical

import autolab_core.utils as utils
from autolab_core import YamlConfig
from perception import Image, RgbdImage
from perception.models.constants import *
from visualization import Visualizer2D as vis

from dexnet.learning import TensorDataset, Tensor

class TrainHistory(Callback):
    """ Callback for saving training history """
    def __init__(self, log_dir, *args, **kwargs):
        self.train_acc_filename = os.path.join(log_dir, 'train_acc.npy')
        self.train_losses_filename = os.path.join(log_dir, 'train_losses.npy')
        self.train_iters_filename = os.path.join(log_dir, 'train_iters.npy')
        self.val_acc_filename = os.path.join(log_dir, 'val_acc.npy')
        self.val_losses_filename = os.path.join(log_dir, 'val_losses.npy')
        self.val_iters_filename = os.path.join(log_dir, 'val_iters.npy')
        Callback.__init__(self, *args, **kwargs)

    def on_train_begin(self, logs={}):
        self.train_acc = []
        self.train_losses = []
        self.train_iters = []
        self.val_acc = []
        self.val_losses = []
        self.val_iters = []

    def on_batch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))
        self.train_losses.append(logs.get('loss'))
        self.train_iters.append(logs.get('batch'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_acc.append(logs.get('val_acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_iters.append(logs.get('batch'))

        np.save(self.train_acc_filename, self.train_acc)
        np.save(self.train_losses_filename, self.train_losses)
        np.save(self.train_iters_filename, self.train_iters)
        np.save(self.val_acc_filename, self.val_acc)
        np.save(self.val_losses_filename, self.val_losses)
        np.save(self.val_iters_filename, self.val_iters)

class TensorDataGenerator(ImageDataGenerator):
    """ A data generator for tensors ."""
    def __init__(self,
                 image_dropout_rate=0.0,
                 image_gaussian_sigma=1e-6,
                 image_gaussian_corrcoef=1e-6,
                 rot_180=False,
                 image_shape=None,
                 data_dropout_rate=0.0,
                 data_gaussian_sigma=1e-6,
                 image_tf_callback=None,
                 image_horiz_flip_callback=None,
                 image_vert_flip_callback=None,
                 num_classes=None,
                 *args, **kwargs):
        ImageDataGenerator.__init__(self, *args, **kwargs)

        if self.zca_whitening:
            raise NotImplementedError('ZCA not available for tensor datasets')

        self.image_dropout_rate = image_dropout_rate
        self.image_gaussian_sigma = image_gaussian_sigma
        self.image_gaussian_corrcoef = image_gaussian_corrcoef
        self.rot_180 = rot_180
        self.image_shape = image_shape

        self.data_dropout_rate = data_dropout_rate
        self.data_gaussian_sigma = data_gaussian_sigma

        self.image_tf_callback = image_tf_callback
        self.image_horiz_flip_callback = image_horiz_flip_callback
        self.image_vert_flip_callback = image_vert_flip_callback

        self.n = None
        self.mean = None
        self.std = None
        self.ssq = None
        self.cov = None
        self.principal_components = None

        self.min_output = None
        self.max_output = None
        self.num_classes = num_classes

    @property
    def resize_images(self):
        return (self.image_shape is not None)

    def standardize(self, x_dict):
        """Apply the normalization configuration to a batch of inputs.

        Parameters
        ----------
        x : :obj:`dict`
            Batch of inputs to be normalized.
        
        Returns
        -------
        :obj:`dict`
            The inputs, normalized.
        """
        for x_name, x in x_dict.iteritems():
            x = x.astype(np.float32)
            num_x = 1
            x0 = x
            if len(x.shape) == 4:
                num_x = x.shape[0]
                if num_x == 0:
                    continue
                x0 = x[0]
                
            if self.preprocessing_function:
                x = self.preprocessing_function(x)
            if self.rescale:
                x *= self.rescale
            # x is a single image, so it doesn't have image number at index 0
            img_channel_axis = self.channel_axis - 1
            if self.samplewise_center:
                x -= np.mean(x, axis=img_channel_axis, keepdims=True)
            if self.samplewise_std_normalization:
                x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

            if self.featurewise_center:
                if self.mean is not None:
                    if Image.can_convert(x0):
                        x -= self.mean[x_name][:x0.shape[2]]
                    else:
                        x -= self.mean[x_name]
                else:
                    warnings.warn('This TensorDataGenerator specifies '
                                  '`featurewise_center`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit()`.')
            if self.featurewise_std_normalization:
                if self.std is not None:
                    if Image.can_convert(x0):
                        x /= (self.std[x_name][:x0.shape[2]] + 1e-7)
                    else:
                        x /= (self.std[x_name] + 1e-7)
                else:
                    warnings.warn('This TensorDataGenerator specifies '
                                  '`featurewise_std_normalization`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit()`.')

            if self.resize_images and Image.can_convert(x0):
                image_shape = x0.shape
                if self.image_shape is not None:
                    image_shape = self.image_shape

                x_resized = np.zeros([num_x] + list(image_shape))
                for i in range(num_x):
                    im = x0
                    if num_x > 1:
                        im = x[i]
                    for c in range(min(image_shape[2], x0.shape[2])):
                        x_resized[i,...,c] = sm.imresize(im[:,:,c],
                                                         size=(image_shape[0],
                                                               image_shape[1]),
                                                         interp='bilinear',
                                                         mode='F')
                    x = x_resized

                x_dict[x_name] = x
        return x_dict

    def random_transform(self, x_dict, seed=None):
        """Randomly augment a single image tensor.
        
        Parameters
        ----------
        x : :obj:`dict`
            dictionary mapping field names to numpy arrays
        seed : float
            random seed

        Returns
        -------
        :obj:`dict`
            a randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # read im height
        im_height = None
        im_width = None
        for x_name in x_dict.keys():
            x = x_dict[x_name]
            if Image.can_convert(x):
                im_height = x.shape[img_row_axis]
                im_width = x.shape[img_col_axis]
            
        # use composition of homographies
        # to generate final transform that needs to be applied
        theta = 0
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        elif self.rot_180:
            if np.random.rand() < 0.5:
                theta = np.pi

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * im_height
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * im_width
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            for x_name in x_dict.keys():
                x = x_dict[x_name]
                if Image.can_convert(x):
                    h, w = x.shape[img_row_axis], x.shape[img_col_axis]
                    tf_matrix = transform_matrix_offset_center(transform_matrix, h, w)
                    x_dict[x_name] = apply_transform(x, tf_matrix, img_channel_axis,
                                                     fill_mode=self.fill_mode, cval=self.cval)
                elif self.image_tf_callback:
                    x_dict[x_name] = self.image_tf_callback(x)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                for x_name in x_dict.keys():
                    x = x_dict[x_name]
                    if Image.can_convert(x):
                        x_dict[x_name] = flip_axis(x, img_col_axis)
                    elif self.image_horiz_flip_callback:
                        x_dict[x_name] = self.image_horiz_flip_callback(x)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                for x_name in x_dict.keys():
                    x = x_dict[x_name]
                    if Image.can_convert(x):
                        x_dict[x_name] = flip_axis(x, img_row_axis)
                    elif self.image_vert_flip_callback:
                        x_dict[x_name] = self.image_vert_flip_callback(x)
        
        for x_name, x in x_dict.iteritems():        
            if Image.can_convert(x):
                image_noise_height = min(x.shape[0], x.shape[0] / self.image_gaussian_corrcoef)
                image_noise_width = min(x.shape[1], x.shape[1] / self.image_gaussian_corrcoef)
                image_noise_channels = x.shape[2]
                image_num_px = image_noise_height * image_noise_width
                for c in range(image_noise_channels):
                    image_noise = ss.norm.rvs(scale=self.image_gaussian_sigma, size=image_num_px)
                    image_noise = image_noise.reshape(image_noise_height, image_noise_width)
                    image_noise = sm.imresize(image_noise, size=float(max(self.image_gaussian_corrcoef, 1)), interp='bilinear', mode='F')
                    x[:,:,c] += image_noise.astype(x.dtype)
            else:
                data_noise = ss.norm.rvs(scale=self.data_gaussian_sigma,
                                         size=x.shape[0])
                x += data_noise.astype(x.dtype)
            x_dict[x_name] = x

        for x_name, x in x_dict.iteritems():
            if Image.can_convert(x):
                num_vals = x.shape[0] * x.shape[1] * x.shape[2]
                num_drop = int(self.image_dropout_rate * num_vals)
                dropout_ind = np.random.choice(num_vals,
                                               size=num_drop)
                dropout_ind = np.unravel_index(dropout_ind, x.shape)
                x[dropout_ind[0], dropout_ind[1], dropout_ind[2]] = 0
            else:
                num_vals = x.shape[0]
                num_drop = int(self.data_dropout_rate * num_vals)
                dropout_ind = np.random.choice(num_vals,
                                               size=num_drop)
                x[dropout_ind] = 0
            x_dict[x_name] = x

        return x_dict

    def flow_from_dataset(self, dataset, x_names, y_name, indices=None, batch_size=32, shuffle=True, seed=None,
                          save_to_dir=None, save_prefix=''):
        return TensorDatasetIterator(
            dataset, x_names, y_name, self,
            indices=indices,
            batch_size=batch_size,
            num_classes=self.num_classes,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix)

    def fit(self, dataset, x_names, y_name,
            indices=None,
            augment=False,
            rounds=1,
            num_tensors=None,
            seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        
        Parameters
        ----------
        dataset : :obj:`TensorDataset`
            The dataset to fit on
        x_names : :obj:`list` of str
            Names of the fields to fit
        y_name :  str
            Name of the classification or regression target
        indices : :obj:`numpy.ndarray`
            Indices of datapoints to use in fitting
        augment : bool
            Whether to fit on randomly augmented samples
        rounds : int
            If `augment`,
            how many augmentation passes to do over the data
        num_tensors : int
            The maximum number of tensors to use for the
            mean and std computation
        seed : float
            Random seed.

        Raises
        ------
        :obj`ValueError`
            in case of invalid input `x`.
        """
        if seed is not None:
            np.random.seed(seed)

        # init bufs
        self.n = {}
        self.mean = {}
        self.std = {}
        self.ssq = {}
        self.cov = {}
        self.principal_components = {}

        # set indices
        if indices is None:
            indices = np.arange(dataset.num_datapoints)

        # sample from the tensor indices
        if num_tensors is None:
            num_tensors = dataset.num_tensors
        num_tensors = min(num_tensors, dataset.num_tensors)
        tensor_indices = np.arange(dataset.num_tensors)
        if num_tensors < dataset.num_tensors:
            np.random.shuffle(tensor_indices)
        tensor_indices = tensor_indices[:num_tensors]

        # compute stats for each input field
        for x_name in x_names:
            logging.info('Fitting %s' %(x_name))

            # init storage
            self.n[x_name] = 0
            self.mean[x_name] = None
            self.std[x_name] = None            
            self.ssq[x_name] = None            
            self.cov[x_name] = None
            self.principal_components[x_name] = None

            # pass #1: compute mean and std using Walford's algorithm
            for i, tensor_ind in enumerate(tensor_indices):
                logging.info('Loading input tensor %d for field %s (%d of %d)' %(tensor_ind, x_name, i+1, num_tensors))

                # load tensor
                x_tensor = dataset.tensor(x_name, tensor_ind)
                datapoint_indices = dataset.datapoint_indices_for_tensor(tensor_ind)
                first_ind = np.min(datapoint_indices)
                last_ind = np.max(datapoint_indices)
                datapoint_indices = indices[(indices >= first_ind) & (indices <= last_ind)]
                datapoint_indices = datapoint_indices - first_ind

                # convert data type and subsample
                x_tensor = Tensor(x_tensor.shape, dtype=K.floatx(),
                                  data=x_tensor.arr[datapoint_indices,...])


                # augment tensor data
                if augment:
                    ax = np.zeros(tuple([rounds * x_tensor.size] + list(x_tensor.shape)[1:]), dtype=K.floatx())            
                    for r in range(rounds):
                        for i, x in enumerate(x_tensor):
                            x_dict = {x_name: x}
                            x_dict = self.random_transform(x_dict)
                            ax[i + r * x_tensor.size] = x_dict[x_name]
                    x_tensor = Tensor(ax.shape, dtype=K.floatx(), data=ax)

                # aggregate mean using online formula
                m_old = np.copy(self.mean[x_name])
                if self.featurewise_center:
                    if self.mean[x_name] is None:
                        if x_tensor.contains_im_data:
                            self.mean[x_name] = np.zeros(x_tensor.channels)
                        else:
                            self.mean[x_name] = np.zeros(x_tensor.height)
                        m_old = np.copy(self.mean[x_name])
                    if x_tensor.contains_im_data:
                        n_tensor = x_tensor.num_datapoints * x_tensor.height * x_tensor.width
                        self.n[x_name] += n_tensor
                        m_new = np.sum(x_tensor.arr, axis=(0, self.row_axis, self.col_axis), dtype=np.float64) / self.n[x_name]
                        m_shrink = n_tensor * m_old / self.n[x_name]
                    else:
                        n_tensor = x_tensor.num_datapoints
                        self.n[x_name] += n_tensor
                        m_new = np.sum(x_tensor.arr, axis=0, dtype=np.float64) / self.n[x_name]
                        m_shrink = n_tensor * m_old / self.n[x_name]
                    self.mean[x_name] += m_new - m_shrink

                # aggregate std using online formula
                # (tracking sum of squared errors)
                if self.featurewise_std_normalization:
                    if self.std[x_name] is None:
                        if x_tensor.contains_im_data:
                            self.std[x_name] = np.zeros(x_tensor.channels)
                            self.ssq[x_name] = np.zeros(x_tensor.channels)
                        else:
                            self.std[x_name] = np.zeros(x_tensor.height)
                            self.ssq[x_name] = np.zeros(x_tensor.height)
                
                    if x_tensor.contains_im_data:
                        self.ssq[x_name] += np.sum((x_tensor.arr - m_old)*(x_tensor.arr - self.mean[x_name]), axis=(0,1,2), dtype=np.float64)
                    else:
                        self.ssq[x_name] += np.sum((x_tensor.arr - m_old)*(x_tensor.arr - self.mean[x_name]), axis=0, dtype=np.float64)

            # take the sqrt to get the true std
            self.std[x_name] = np.sqrt(self.ssq[x_name] / (self.n[x_name] - 1))
                        
            # pass #2: compute zca
            if self.zca_whitening:
                raise NotImplementedError('ZCA not available for tensor datasets')

        # compute stats for each output field
        self.min_output = np.inf
        self.max_output = -np.inf
        for i, tensor_ind in enumerate(tensor_indices):
            logging.info('Loading output tensor %d for field %s (%d of %d)' %(tensor_ind, y_name, i+1, num_tensors))

            # load tensor
            y_tensor = dataset.tensor(y_name, tensor_ind)
        
            # aggregate stats
            self.min_output = min(self.min_output, np.min(y_tensor.arr))
            self.max_output = max(self.max_output, np.max(y_tensor.arr))
        if self.num_classes is None:
            self.num_classes = int(self.max_output+1)
            
class TensorDatasetIterator(Iterator):
    """Iterator yielding data from a tensor dataset.

    # Arguments
        x_name: Field names for input data.
        y_name: Field name for targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
    """
    def __init__(self, dataset, x_names, y_name, data_generator,
                 indices=None, batch_size=32, num_classes=1, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix=''):
        for x_name in x_names:
            if x_name not in dataset.field_names:
                raise ValueError('Input field name %s not in dataset!' %(x_name)) 
        if y_name not in dataset.field_names:
            raise ValueError('Target field name %s not in dataset!' %(y_name))

        self.dataset = dataset
        self.x_names = x_names
        self.y_name = y_name
        self.indices = indices
        self.batch_size = batch_size
        self.num_classes = num_classes
        if data_format is None:
            data_format = K.image_data_format()
        self.data_generator = data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix

        if self.indices is None:
            self.indices = np.arange(self.dataset.num_datapoints)

        self.all_names = self.x_names + [self.y_name]
        self.iter_batch_size = batch_size
        self.tensors_per_batch = int(np.ceil(float(batch_size) / dataset.datapoints_per_file))
        super(TensorDatasetIterator, self).__init__(dataset.num_tensors, self.tensors_per_batch, shuffle, seed)

    def _preprocess_input(self, x):
        """ Preprocesses input data. """
        x = self.data_generator.random_transform(x)
        x = self.data_generator.standardize(x)
        return x

    def _get_batches_of_transformed_samples(self, index_array):
        """ Yields a batch of transformed samples. """
        # allocate new datapoint
        batch_x = {}
        for x_name in self.x_names:
            x_shape = list(self.dataset.tensors[x_name].shape[1:])
            if Image.can_convert(self.dataset.tensors[x_name][0]) and \
               self.data_generator.resize_images:
                x_shape = self.data_generator.image_shape
            x_shape = [self.iter_batch_size] + list(x_shape)
            x_dtype = self.dataset.tensors[x_name].dtype
            batch_x[x_name] = np.zeros(x_shape, dtype=np.float32)
        y_shape = [self.iter_batch_size, self.num_classes]
        y_dtype = self.dataset.tensors[self.y_name].dtype
        batch_y = np.zeros(y_shape, dtype=y_dtype)

        # iteratively load sampled tensors
        batch_ind = 0
        for tensor_ind in index_array[0]:
            # compute num remaining
            num_remaining = self.iter_batch_size - batch_ind

            # select datapoint indices within the tensor
            datapoint_indices = self.dataset.datapoint_indices_for_tensor(tensor_ind)
            first_ind = np.min(datapoint_indices)
            last_ind = np.max(datapoint_indices)
            datapoint_indices = self.indices[(self.indices >= first_ind) & (self.indices <= last_ind)]
            num_datapoints = datapoint_indices.shape[0]
            num_to_sample = min(num_datapoints, num_remaining)
            indices = np.random.choice(datapoint_indices,
                                       size=num_to_sample)

            # preprocess
            for i, datapoint_ind in enumerate(indices):
                x = self.dataset.datapoint(datapoint_ind,
                                           field_names=self.x_names)
                y = self.dataset.datapoint(datapoint_ind,
                                           field_names=[self.y_name])
                x = self._preprocess_input(x)
                for x_name in self.x_names:
                    batch_x[x_name][batch_ind,...] = x[x_name]
                batch_y[batch_ind,...] = to_categorical(y[self.y_name],
                                                        num_classes=self.num_classes)
                batch_ind += 1

        # subsample data
        for x_name in self.x_names:
            batch_x[x_name] = batch_x[x_name][:batch_ind,...]
        batch_y = batch_y[:batch_ind]

        # optionally, save images
        if self.save_to_dir:
            for x_name, x_tensor in batch_x.iteritems():
                if Image.can_convert(x_tensor[0]):
                    for i, x in enumerate(x_tensor):
                        filename = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                             index=i,
                                                                             hash=np.random.randint(1e4),
                                                                             format='npy')
                        np.save(filename, x)

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

