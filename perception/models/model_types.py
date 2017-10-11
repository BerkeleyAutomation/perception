"""
Available CNN model types.
Author: Jeff Mahler
"""

from perception.models import ClassificationCNN, FinetunedClassificationCNN, ResNet50, VGG16

CLASSIFICATION_CNN_TYPES = {
    'finetuned': FinetunedClassificationCNN,
    'resnet50': ResNet50,
    'vgg16': VGG16
}
