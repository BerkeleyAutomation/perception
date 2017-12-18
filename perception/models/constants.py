"""
Constants for deep neural network models
Author: Jeff Mahler
"""
import numpy as np

IMAGENET_BGR_MEAN = np.array([103.939, 116.779, 123.68])
IMAGENET_DEFAULT_SIZE = 224
IMAGENET_MIN_SIZE = 48
IMAGENET_NUM_CLASSES = 1000
IMAGENET_WEIGHT_TYPE = 'imagenet'
IMAGENET_INPUT_NAME = 'image'
IMAGENET_OUTPUT_NAME = 'label'

VGG_OUTPUT_NAME = 'predictions'

RESNET_OUTPUT_NAME = 'fc1000'

ALEXNET_OUTPUT_NAME = 'softmax'
