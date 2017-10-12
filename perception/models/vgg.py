"""
Keras implementation of VGG-16.
Modified from keras.applications.vgg16
Author: Jeff Mahler
"""
import cv2
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings

from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from perception.models.constants import *
from perception.models import ClassificationCNN

class VGG16(ClassificationCNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a VGG-16 model.
        """
        kwargs['output_name'] = VGG_OUTPUT_NAME
        ClassificationCNN.__init__(self, *args, **kwargs)

    def _build_network(self, input_tensor=None, include_fc=True,
                       output_pooling=None, output_name=VGG_OUTPUT_NAME):
        """ Build the VGG-16 network """
        # Set inputs
        if input_tensor is None:
            input_tensor = self._input_tensor

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(self._input_tensor)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if include_fc:
            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(self._num_classes, activation='softmax', name=VGG_OUTPUT_NAME)(x)
        else:
            if output_pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif output_pooling == 'max':
                x = GlobalMaxPooling2D()(x)
        
        return x

