"""
Test implementation of VGG-16
Author: Jeff Mahler
"""
import cv2
import IPython
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from perception.models import VGG16

DEFAULT_VGG16_WEIGHTS = '/home/autolab/Public/data/dex-net/data/models/classification/vgg16/weights.h5'

if __name__ == '__main__':
    image_filename = sys.argv[1]

    with open('data/images/imagenet.json', 'r') as f:
        label_to_category = eval(f.read())

    im = cv2.imread(image_filename)
    im = cv2.resize(im, (224, 224)).astype(np.float32)
    vgg = VGG16(weights_filename=DEFAULT_VGG16_WEIGHTS)
    out = vgg.predict(im)
    label = vgg.top_prediction(im)
    category = label_to_category[label]

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(im.astype(np.uint8))
    plt.title('Pred: %s' %(category))
    plt.show()

