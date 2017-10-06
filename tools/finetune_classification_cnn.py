"""
Finetunes a CNN for classification on a custom dataset using keras
Author: Jeff Mahler
"""
import logging
import IPython
import numpy as np
import os
import sys
import time

import scipy.misc as sm
import scipy.stats as ss

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, transform_matrix_offset_center, apply_transform

from autolab_core import YamlConfig
from perception import Image, RgbdImage
from perception.models import ResNet50
from visualization import Visualizer2D as vis

from dexnet.learning import TensorDataset, Tensor

class TensorDataGenerator(ImageDataGenerator):
    """ A data generator for tensors ."""
    def __init__(self,
                 image_dropout_rate=0.0,
                 image_gaussian_sigma=1e-6,
                 image_gaussian_corrcoef=1e-6,
                 rot_180=False,
                 data_dropout_rate=0.0,
                 data_gaussian_sigma=1e-6,
                 image_tf_callback=None,
                 image_horiz_flip_callback=None,
                 image_vert_flip_callback=None,
                 *args, **kwargs):
        ImageDataGenerator.__init__(self, *args, **kwargs)

        if self.zca_whitening:
            raise NotImplementedError('ZCA not available for tensor datasets')

        self.image_dropout_rate = image_dropout_rate
        self.image_gaussian_sigma = image_gaussian_sigma
        self.image_gaussian_corrcoef = image_gaussian_corrcoef
        self.rot_180 = rot_180

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
                    x -= self.mean[x_name]
                else:
                    warnings.warn('This TensorDataGenerator specifies '
                                  '`featurewise_center`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit()`.')
            if self.featurewise_std_normalization:
                if self.std is not None:
                    x /= (self.std[x_name] + 1e-7)
                else:
                    warnings.warn('This TensorDataGenerator specifies '
                                  '`featurewise_std_normalization`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit()`.')
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

        # use composition of homographies
        # to generate final transform that needs to be applied
        theta = 0
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        elif self.rot_180:
            if np.random.rand() < 0.5:
                theta = np.pi

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
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
                    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
                    x_dict[x_name] = apply_transform(x, transform_matrix, img_channel_axis,
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
                    x[:,:,c] += image_noise
            else:
                data_noise = ss.norm.rvs(scale=self.data_gaussian_sigma,
                                         size=x.shape[0])
                x += data_noise
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

    def flow_from_dataset(self, dataset, x_names, y_name, batch_size=32, shuffle=True, seed=None,
                          save_to_dir=None, save_prefix='', save_format='png'):
        return TensorDatasetIterator(
            dataset, x_names, y_name, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def fit(self, dataset, x_names,
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

        # sample from the tensor indices
        if num_tensors is None:
            num_tensors = dataset.num_tensors
        tensor_indices = np.arange(dataset.num_tensors)
        if num_tensors < dataset.num_tensors:
            np.random.shuffle(tensor_indices)
        tensor_indices = tensor_indices[:num_tensors]

        # compute stats for each field
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
                logging.info('Loading tensor %d (%d of %d)' %(tensor_ind, i+1, num_tensors))

                # load tensor
                x_tensor = dataset.tensor(x_name, tensor_ind)

                # convert data type
                x_tensor = Tensor(x_tensor.shape, dtype=K.floatx(), data=x_tensor.arr)

                # augment tensor data
                if augment:
                    ax = np.zeros(tuple([rounds * x_tensor.size] + list(x_tensor.shape)[1:]), dtype=K.floatx())            
                    for r in range(rounds):
                        for i, x in enumerate(x_tensor):
                            logging.info('Transforming datapoint %d' %(i + r * x_tensor.size))
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
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """
    def __init__(self, dataset, x_names, y_name, data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        for x_name in x_names:
            if x_name not in dataset.field_names:
                raise ValueError('Input field name %s not in dataset!' %(x_name)) 
        if y_name not in dataset.field_names:
            raise ValueError('Target field name %s not in dataset!' %(y_name))

        self.dataset = dataset
        self.x_names = x_names
        self.y_name = y_name
        self.batch_size = batch_size
        if data_format is None:
            data_format = K.image_data_format()
        self.data_generator = data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        tensors_per_batch = 1 + batch_size / dataset.datapoints_per_file
        super(TensorDatasetIterator, self).__init__(dataset.num_tensors, tensors_per_batch, shuffle, seed)

    def _preprocesses_input(x_dict):
        """ Preprocesses input data. """
        x_dict = self.data_generator.random_transform(x_dict)
        x_dict = self.data_generator.standardize(x_dict)
        return x_dict

    def _get_batches_of_transformed_samples(self, index_array):
        """ Yields a batch of transformed samples. """
        batch_x = {}
        for x_name in self.x_names:
            x_shape = [self.batch_size] + self.dataset.tensors[x_name].shape
            x_dtype = self.dataset.tensors[x_name].dtype
            batch_x[x_name] = Tensor(x_shape, x_dtype)
        y_shape = [self.batch_size]
        y_dtype = self.dataset.tensors[self.y_name].dtype
        batch_y = Tensor(y_shape, y_dtype)

        for tensor_ind in index_array:
            num_queued = batch_y.size
            num_remaining = self.batch_size - num_queued
            
            num_to_sample = min(self.batch_size, num_remaining)
            indices = np.random.choice(x_tensor.num_datapoints, size=num_to_sample)

            x_dicts = [{}] * num_to_sample
            for x_name in self.x_names:
                x_tensor = self.dataset.tensor(x_name, tensor_ind)
                for j, index in enumerate(indices):
                    x_dicts[j][x_name] = x_tensor.data[index,...]
                    
            for x_dict in x_dicts:
                self._preprocess_input(x_dict)
                for x_name, x in x_dict.iteritems():
                    batch_x[x_name].add(x)

            y_tensor = self.dataset.tensor(self.y_name, tensor_ind)                   
            batch_y.add_batch(y_tensor.data[indices,...])

        if self.save_to_dir:
            for x_name, x_tensor in batch_x.iteritems():
                if x_tensor.contains_im_data:
                    for i, x in enumerate(x_tensor):
                        im = Image.from_array(x)
                        filename = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                             index=i,
                                                                             hash=np.random.randint(1e4),
                                                                             format=self.save_format)
                        im.save(filename)
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

def finetune_network(config):
    """ Main function. """
    # read params
    x_names = config['x_names']
    y_name = config['y_name']
    dataset_dir = config['dataset']

    # data augmentation params
    data_aug_config = config['data_augmentation']

    # preprocessing params
    preproc_config = config['preprocessing']

    # create train and test dataset names
    train_dataset_dir = os.path.join(dataset_dir, 'train')
    val_dataset_dir = os.path.join(dataset_dir, 'val')

    # read in the dataset
    train_dataset = TensorDataset.open(train_dataset_dir)
    val_dataset = TensorDataset.open(val_dataset_dir)

    # create train and test generators
    train_generator = TensorDataGenerator(**data_aug_config)
    train_generator.fit(train_dataset, x_names, **preproc_config)
   
    test_generator = TensorDataGenerator(featurewise_center=featurewise_center,
                                          featurewise_std_normalization=featurewise_std_normalization,
                                          rotation_range=rotation_range,
                                          image_gaussian_sigma=image_gaussian_sigma,
                                          image_gaussian_corrcoef=image_gaussian_corrcoef)
    test_generator.fit(test_dataset, x_names,
                        augment=augment_mean_std_fit,
                        rounds=num_augmentations)


def test_mean_std(config):
    """ Main function. """
    # read params
    dataset = config['dataset']
    x_names = config['x_names']
    y_name = config['y_name']

    data_aug_config = config['data_augmentation']
    preproc_config = config['preprocessing']

    # open dataset
    dataset = TensorDataset.open(dataset)

    # generator
    generator = TensorDataGenerator(**data_aug_config)
    fit_start = time.time()
    generator.fit(dataset, x_names, **preproc_config)
    fit_stop = time.time()
    logging.info('Generator fit took %.3f sec' %(fit_stop - fit_start))

    # compute mean and std
    logging.info('Computing mean and std with brute force')
    fit_start = time.time()
    im = False
    x = None
    cur_i = 0
    end_i = dataset.datapoints_per_file
    for ind in range(dataset.num_tensors):
        tensor = dataset.tensor(x_names[0], ind)
        if x is None:
            if tensor.width is not None:
                x = np.zeros([dataset.num_datapoints, tensor.height,
                              tensor.width, tensor.channels])
                im = True
            else:
                x = np.zeros([dataset.num_datapoints, tensor.height])
        x[cur_i:end_i,...] = tensor.data
        cur_i = end_i
        end_i = cur_i + dataset.datapoints_per_file

    if im:
        mean_brute = np.mean(x, axis=(0,1,2), dtype=np.float64)
        std_brute = np.std(x, axis=(0,1,2), dtype=np.float64)
    else:
        mean_brute = np.mean(x, axis=0, dtype=np.float64)
        std_brute = np.std(x, axis=0, dtype=np.float64)

    fit_stop = time.time()
    logging.info('Brute force fit took %.3f sec' %(fit_stop - fit_start))

    print 'GENERATOR'
    print 'Mean:', generator.mean[x_names[0]]
    print 'Std:', generator.std[x_names[0]]
    print 'BRUTE'
    print 'Mean:', mean_brute
    print 'Std:', std_brute

    logging.info('Test succeeded!')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # read config
    config = YamlConfig(sys.argv[1])

    # finetune
    test_mean_std(config)
