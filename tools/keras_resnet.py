"""
Test implementation of ResNet-50
Author: Jeff Mahler
"""
import cv2
import IPython
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from perception import ColorImage
from perception.models import ResNet50

DEFAULT_RESNET50_WEIGHTS = '/home/autolab/Public/data/dex-net/data/models/classification/resnet50/weights.h5'

if __name__ == '__main__':
    image_filename = sys.argv[1]

    with open('data/images/imagenet.json', 'r') as f:
        label_to_category = eval(f.read())

    im = ColorImage.open(image_filename)
    resnet = ResNet50(weights_filename=DEFAULT_RESNET50_WEIGHTS)
    out = resnet.predict(im)
    label = resnet.top_prediction(im)
    category = label_to_category[label]

    plt.figure()
    plt.imshow(im.bgr2rgb().data)
    plt.title('Pred: %s' %(category))
    plt.show()

    IPython.embed()
