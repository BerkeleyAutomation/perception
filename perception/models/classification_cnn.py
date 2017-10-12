"""
CNN-based Classifier base class
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import copy
import cPickle as pkl
import cv2
import IPython
import logging
import numpy as np
import os
import sys
import warnings
import yaml

import scipy.misc as sm

from keras.engine.topology import get_source_inputs
from keras.layers import Input, Reshape, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape

from autolab_core import YamlConfig
from perception.models.constants import *
from perception.models import TrainHistory, TensorDataGenerator, TensorDatasetIterator

class ClassificationCNN(object):
    """ Base class for CNN-based classification.
    """
    def __init__(self, include_fc=True,
                 input_tensor=None,
                 input_shape=None,
                 input_name=IMAGENET_INPUT_NAME,
                 output_name=IMAGENET_OUTPUT_NAME,
                 output_pooling=None,
                 im_preprocessor=None,
                 num_classes=IMAGENET_NUM_CLASSES,
                 name='classification',
                 weights_type=IMAGENET_WEIGHT_TYPE,
                 weights_filename=None, **kwargs):
        """
        Initialize a CNN model.
        """
        # check input weights
        if weights_filename is not None and not os.path.exists(weights_filename):
            raise ValueError('Weights filename %s does not exist!' %(weights_filename))

        # read params
        self._num_classes = num_classes
        self._include_fc = include_fc

        # setup image preprocessing
        self._im_preprocessor = im_preprocessor
        if im_preprocessor is None:
            self._im_preprocessor = ImageDataGenerator(featurewise_center=True)
            self._im_preprocessor.mean = IMAGENET_BGR_MEAN

        # get input shape
        self._input_shape = _obtain_input_shape(input_shape,
                                                default_size=IMAGENET_DEFAULT_SIZE,
                                                min_size=IMAGENET_MIN_SIZE,
                                                data_format=K.image_data_format(),
                                                require_flatten=True,
                                                weights=weights_type)

        # build input tensor
        if input_tensor is None:
            self._input_tensor = Input(shape=self.input_shape, name=input_name)
        else:
            if not K.is_keras_tensor(input_tensor):
                self._input_tensor = Input(tensor=input_tensor, shape=self.input_shape,
                                           name=input_name)
            else:
                self._input_tensor = input_tensor
            
        # build output tensor
        self._output_tensor = self._build_network(include_fc=include_fc,
                                                  output_pooling=output_pooling,
                                                  output_name=output_name)

        # set names
        self._input_name = self.input_tensor.name[:self.input_tensor.name.rfind(':')]
        self._output_name = self.output_tensor.name[:self.output_tensor.name.rfind(':')]

        # build standalone model
        self._model = self._build_model(input_tensor=input_tensor, name=name)

        # optionally auto-load weights
        if weights_filename is not None:
            self.model.load_weights(weights_filename)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def input_tensor(self):
        return self._input_tensor

    @property
    def output_tensor(self):
        return self._output_tensor

    @property
    def im_height(self):
        return self.input_tensor.shape[1].value

    @property
    def im_width(self):
        return self.input_tensor.shape[2].value

    @property
    def channels(self):
        return self.input_tensor.shape[3].value

    @property
    def model(self):
        return self._model

    @property
    def layers(self):
        return self.model.layers

    @property
    def include_fc(self):
        return self._include_fc

    @property
    def input(self):
        return self.model.input

    @property
    def input_name(self):
        return self._input_name

    @property
    def output(self):
        return self.model.output

    @property
    def output_name(self):
        return self._output_name

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def im_preprocessor(self):
        return self._im_preprocessor

    @property
    def config(self):
        from perception.models.model_types import CLASSIFICATION_CNN_TYPENAMES
        config = {}
        for key, val in self.__dict__.iteritems():
            if type(val) in (int, float, bool, str, dict, list, tuple, unicode):
                k = copy.copy(key)
                if k[0] == '_':
                    k = k[1:]
                config[k] = val
        config['model_type'] = CLASSIFICATION_CNN_TYPENAMES[type(self)]
        return config

    @abstractmethod
    def _build_network(self, input_tensor=None, include_fc=True,
                       output_pooling=None, output_name=IMAGENET_OUTPUT_NAME):
        """ Build the network """
        pass

    def _build_model(self, input_tensor=None, name='classifier'):
        """ Build the model. """
        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        inputs = self.input_tensor
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        # Create model.
        model = Model(inputs, self.output_tensor, name=name)
        return model

    def predict(self, im):
        """ Predict the classwise probabilities for a single image. """
        # resize image
        pred_im = im.resize((self.im_height, self.im_width))
        pred_im_arr = pred_im.raw_data.astype(np.float32)

        # preprocess
        pred_im_arr = self.im_preprocessor.standardize(pred_im_arr)
        pred_im_arr = np.expand_dims(pred_im_arr, axis=0)

        # predict
        return self.model.predict(pred_im_arr)

    def top_prediction(self, im):
        """ Predict the most likely class for a single image. """
        return np.argmax(self.predict(im))

    def evaluate_on_dataset(self, dataset, indices=None, batch_size=128):
        """ Evaluate predictions and true labels on
        a dataset for the subset of indices. """
        input_name = self.input_name
        output_name = self.output_name[:self.output_name.find('/')]
        if indices is None:
            indices = np.arange(dataset.num_datapoints)

        iterator = self.im_preprocessor.flow_from_dataset(dataset,
                                                          [input_name],
                                                          output_name,
                                                          indices=indices,
                                                          shuffle=False,
                                                          batch_size=batch_size)

        num_predict = indices.shape[0]
        pred_probs = np.zeros([num_predict, self.num_classes])
        labels = np.zeros(num_predict)
        cur_i = 0
        for i in range(dataset.num_tensors):
            logging.info('Predicting batch %d' %(i))
            batch_x, batch_y = iterator.next()
            batch_size = batch_y.shape[0]
            end_i = cur_i + batch_size
            pred_probs[cur_i:end_i,:] = self.model.predict_on_batch(batch_x[input_name])
            labels[cur_i:end_i] = np.argmax(batch_y, axis=1)
            cur_i = end_i
        return pred_probs, labels

    def predict_batch_probs(self, im_arr):
        """ Predict the classwise probabilities for an array of images. """
        # check image size
        if len(im_arr.shape) != 4:
            raise ValueError('Must provide a 4-channel tensor!')
        # TODO: add back in
        #if im_arr.shape[3] != self.channels:
        #    raise ValueError('Number of input channels (%d) does not match expected number of input channels (%d)!' %(im_arr.shape[3], self.channels))
            
        # preprocess
        im_arr_dict = {self.input_name: im_arr.astype(np.float32)}
        im_arr_dict = self.im_preprocessor.standardize(im_arr_dict)

        # resize images
        im_arr = im_arr_dict[self.input_name][:,:,:3] # TODO:remove
        if im_arr.shape[1] != self.im_height or im_arr.shape[2] != self.im_width:
            new_im_arr = np.zeros([im_arr.shape[0], self.im_height, self.im_width, self.channels])
            for i, im in enumerate(im_arr):
                new_im_arr[i,...] = sm.imresize(im,
                                                size=(self.im_height,
                                                      self.im_width),
                                                interp='bilinear')
            im_arr = new_im_arr

        # predict
        return self.model.predict(im_arr)

    def predict_batch_labels(self, im_arr):
        """ Predict the most likely class for an array of images. """
        return np.argmax(self.predict_batch_probs(im_arr), axis=1)

    def load(self, weights_filename):
        """ Load a set of weights. """
        self.model.load_weights(weights_filename)            

    def save(self, model_dir):
        """ Save a model to disk. """
        # create dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        # save weights
        weights_filename = os.path.join(model_dir, 'weights.h5')
        self.model.save(weights_filename)

        # save im preprocessor
        im_preprocessor_filename = os.path.join(model_dir, 'preprocessor.pkl')
        pkl.dump(self.im_preprocessor, open(im_preprocessor_filename, 'wb'))

        # save config
        config_filename = os.path.join(model_dir, 'config.yaml')
        yaml.dump(self.config, open(config_filename, 'w'))

    @staticmethod
    def open(model_dir, model_typename=None, *args, **kwargs):
        """ Opens a CNN model from a specified model directory.
        """
        # import other model types
        from perception.models import CLASSIFICATION_CNN_TYPES

        # form filenames
        weights_filename = os.path.join(model_dir, 'weights.h5')
        im_preprocessor_filename = os.path.join(model_dir, 'preprocessor.pkl')
        config_filename = os.path.join(model_dir, 'config.yaml')

        # load preprocessor
        im_preprocessor = None
        if os.path.exists(im_preprocessor_filename):
            im_preprocessor = pkl.load(open(im_preprocessor_filename, 'rb'))
        
        # load params
        config = {}
        if os.path.exists(config_filename):
            config = YamlConfig(config_filename)
        config.update(kwargs)
        if model_typename is None:
            model_typename = config['model_type']

        # check typename
        if model_typename not in CLASSIFICATION_CNN_TYPES.keys():
            raise ValueError('Model type %s not supported!' %(model_typename))

        # create class
        model_type = CLASSIFICATION_CNN_TYPES[model_typename]
        if model_type == FinetunedClassificationCNN:
            base_config_filename = os.path.join(model_dir, 'base_config.yaml')
            base_config = YamlConfig(base_config_filename)
            base_cnn_typename = base_config['model_type']
            base_type = CLASSIFICATION_CNN_TYPES[base_cnn_typename]
            base_cnn = base_type(**base_config)
            return FinetunedClassificationCNN(*args,
                                              base_cnn=base_cnn,
                                              weights_filename=weights_filename,
                                              im_preprocessor=im_preprocessor,
                                              **config)
        return model_type(*args,
                          weights_filename=weights_filename,
                          im_preprocessor=im_preprocessor,
                          **config)

class FinetunedClassificationCNN(ClassificationCNN):
    """ A classification CNN that augments an existing architecture.
    """
    def __init__(self, base_cnn,
                 num_classes=IMAGENET_NUM_CLASSES,
                 new_fc_layer_sizes=[],
                 activation='relu',
                 *args,
                 **kwargs):
        # set base cnn
        self._base_cnn = base_cnn

        # read fc params
        self._new_fc_layer_sizes = new_fc_layer_sizes
        self._activation = activation

        # load super class
        kwargs['input_tensor'] = self.base_cnn.output
        kwargs['input_shape'] = self.base_cnn.input_shape
        kwargs['input_name'] = self.base_cnn.input_name
        kwargs['num_classes'] = num_classes
        ClassificationCNN.__init__(self, *args, **kwargs)
        self._input_tensor = self.base_cnn.input
        self._input_name = self.input_tensor.name[:self.input_tensor.name.rfind(':')]

    @property
    def base_cnn(self):
        """ Returns the base CNN object. """
        return self._base_cnn

    @property
    def num_new_fc_layers(self):
        return len(self._new_fc_layer_sizes) + 1

    @property
    def new_fc_layer_sizes(self):
        return self._new_fc_layer_sizes

    @property
    def activation(self):
        return self._activation

    def freeze_base_cnn(self):
        for layer in self.base_cnn.model.layers:
            layer.trainable = False

    def _build_network(self, input_tensor=None, include_fc=True,
                       output_pooling=None, output_name=IMAGENET_OUTPUT_NAME):
        """ Build the network """
        return self._add_fc_layers(input_tensor=input_tensor,
                                   output_pooling=output_pooling,
                                   output_name=output_name)

    def _add_fc_layers(self, input_tensor=None, output_pooling=None,
                       output_name=IMAGENET_OUTPUT_NAME):
        """ Adds FC layers to the input tensor. """
        # set inputs
        if input_tensor is None:
            input_tensor = self.base_cnn.output_tensor

        # check the existence of new fc layers
        if self.num_new_fc_layers == 0:
            return input_tensor

        # set intermediate FC layers
        x = Reshape((-1,))(input_tensor)
        for i in range(self.num_new_fc_layers-1):
            layer_name = 'augmented_fc_%03d' %(i)
            x = Dense(self.new_fc_layer_sizes[i], activation=self.activation, name=layer_name)(x)

        # add last output layer
        x = Dense(self.num_classes, activation='softmax', name=output_name)(x)
        return x

    def save(self, model_dir):
        """ Save a model to disk. """
        # save the original CNN model
        ClassificationCNN.save(self, model_dir)

        # save config
        config_filename = os.path.join(model_dir, 'base_config.yaml')
        yaml.dump(self.base_cnn.config, open(config_filename, 'w'))
        

