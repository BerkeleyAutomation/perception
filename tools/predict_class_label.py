"""
Test implementation of CNN classifiers
Author: Jeff Mahler
"""
import cv2
import IPython
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from perception import ColorImage
from perception.models import ClassificationCNN

if __name__ == '__main__':
    model_dir = sys.argv[1]
    model_type = sys.argv[2]
    image_filename = sys.argv[3]

    #with open('data/images/imagenet.json', 'r') as f:
    #    label_to_category = eval(f.read())

    im = ColorImage.open(image_filename)
    cnn = ClassificationCNN.open(model_dir, model_typename=model_type)
    out = cnn.predict(im)
    label = cnn.top_prediction(im)
    #category = label_to_category[label]

    plt.figure()
    plt.imshow(im.bgr2rgb().data)
    plt.title('Pred: %d' %(label))
    plt.axis('off')
    plt.show()

    #IPython.embed()
