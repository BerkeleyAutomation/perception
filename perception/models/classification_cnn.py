"""
CNN-based Classifier base class
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import copy
import cPickle as pkl
import cv2
import IPython
import numpy as np
import os
import sys
import warnings
import yaml

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

class ClassificationCNNType(object):
    """ Enum for different classification CNNs. """
    ResNet50 = 'resnet50'
    VGG16 = 'vgg16'
    Finetuned = 'finetuned'
    Generic = 'generic'

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
                 weights_filename=None):
        """
        Initialize a CNN model.
        """
        # check input weights
        if weights_filename is not None and not os.path.exists(weights_filename):
            raise ValueError('Weights filename %s does not exist!' %(weights_filename))

        # read params
        self._num_classes = num_classes

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
        self._input_name = self.input_tensor.name
        self._output_name = self.output_tensor.name

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
    def model(self):
        return self._model

    @property
    def layers(self):
        return self.model.layers

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
        config = {}
        for key, val in self.__dict__.iteritems():
            if type(val) in (int, float, bool, str, dict, list, tuple, unicode):
                k = copy.copy(key)
                if k[0] == '_':
                    k = k[1:]
                config[k] = val
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

        # subtract mean
        pred_im_arr = self.im_preprocessor.standardize(pred_im_arr)
        pred_im_arr = np.expand_dims(pred_im_arr, axis=0)

        # predict
        return self.model.predict(pred_im_arr)

    def top_prediction(self, im):
        """ Predict the most likely class for a single image. """
        return np.argmax(self.predict(im))

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

        # check typename
        if model_typename not in CLASSIFICATION_CNN_TYPES.keys():
            raise ValueError('Model type %s not supported!' %(model_type))

        # create class
        model_type = CLASSIFICATION_CNN_TYPES[model_typename]
        if model_type == FinetunedClassificationCNN:
            base_cnn_type = config['base_cnn_typename']
            base_type = CLASSIFICATION_CNN_TYPES[base_cnn_typename]
            base_config_filename = os.path.join(model_dir, 'base_config.yaml')
            base_config = YamlConfig(base_config_filename)
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
        ClassificationCNN.__init__(self, input_tensor=self.base_cnn.output,
                                   input_shape=self.base_cnn.input_shape,
                                   input_name=self.base_cnn.input_name,
                                   num_classes=num_classes,
                                   *args,
                                   **kwargs)

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
        

