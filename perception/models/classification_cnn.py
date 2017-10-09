"""
CNN-based Classifier base class
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import cv2
import IPython
import numpy as np
import os
import sys
import warnings

from keras.engine.topology import get_source_inputs
from keras.layers import Input
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape

from perception.models.constants import *

class ClassificationCNN(object):
    """ Base class for CNN-based classification. """
    def __init__(self, weights_filename=None, include_fc=True,
                 input_tensor=None, input_shape=None,
                 output_pooling=None,
                 im_mean=IMAGENET_BGR_MEAN,
                 num_classes=IMAGENET_NUM_CLASSES):
        """
        Initialize a CNN model.
        """
        # check input weights
        if weights_filename is not None and not os.path.exists(weights_filename):
            raise ValueError('Weights filename %s does not exist!' %(weights_filename))

        # read params
        self._num_classes = num_classes
        self._im_mean = im_mean

        # get input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=IMAGENET_DEFAULT_SIZE,
                                          min_size=IMAGENET_MIN_SIZE,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_fc,
                                          weights=weights_filename)

        # build input tensor
        if input_tensor is None:
            self._input_tensor = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                self._input_tensor = Input(tensor=input_tensor, shape=input_shape)
            else:
                self._input_tensor = input_tensor
            
        # build output tensor
        self._output_tensor = self._build_network(include_fc=include_fc,
                                                  output_pooling=output_pooling)

        # build standalone model
        self._model = self._build_model(input_tensor)

        # load weights
        if weights_filename is not None:
            self._model.load_weights(weights_filename)

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
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def im_mean(self):
        return self._im_mean

    @abstractmethod
    def _build_network(self, input_tensor=None, include_fc=True,
                       output_pooling=None):
        """ Build the network """
        pass

    def _build_model(self, input_tensor=None):
        """ Build the model. """
        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        inputs = self.input_tensor
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        # Create model.
        model = Model(inputs, self.output_tensor, name='vgg16')
        return model

    def predict(self, im):
        """ Predict the classwise probabilities for a single image. """
        pred_im = np.copy(im)
        
        # resize image
        pred_im = cv2.resize(pred_im, (self.im_height, self.im_width)).astype(np.float32)

        # subtract mean
        pred_im[:,:,0] -= self.im_mean[0]
        pred_im[:,:,1] -= self.im_mean[1]
        pred_im[:,:,2] -= self.im_mean[2]
        pred_im = np.expand_dims(pred_im, axis=0)

        # predict
        return self.model.predict(pred_im)

    def top_prediction(self, im):
        """ Predict the most likely class for a single image. """
        return np.argmax(self.predict(im))
