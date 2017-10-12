"""
Keras implementation of ResNet-50.
Modified from keras.applications.resnet50
Author: Jeff Mahler
"""
import cv2
import IPython
import logging
import numpy as np
import os
import sys
import warnings

from keras.layers import Flatten, Dense, Input, Conv2D, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras import backend as K
from keras.applications.resnet50 import identity_block, conv_block

from perception.models.constants import *
from perception.models import ClassificationCNN

class ResNet50(ClassificationCNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a ResNet-50 model.
        """
        if K.image_data_format() == 'channels_last':
            self._bn_axis = 3
        else:
            self._bn_axis = 1
        kwargs['output_name'] = RESNET_OUTPUT_NAME
        ClassificationCNN.__init__(self, *args, **kwargs)

    @property
    def bn_axis(self):
        return self._bn_axis
        
    def _build_network(self, input_tensor=None, include_fc=True,
                       output_pooling=None, output_name=None):
        """ Build the ResNet-50 network """
        # Set inputs
        if input_tensor is None:
            input_tensor = self._input_tensor

        x = Conv2D(
            64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_tensor)
        x = BatchNormalization(axis=self.bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
        
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        
        if include_fc:
            x = Flatten()(x)
            x = Dense(self.num_classes, activation='softmax', name=RESNET_OUTPUT_NAME)(x)
        else:
            if output_pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif output_pooling == 'max':
                x = GlobalMaxPooling2D()(x)            
        return x
