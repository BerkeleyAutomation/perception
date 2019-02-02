import numpy as np
import imutils
import os
import cv2
import random
import uuid
import scipy.ndimage
from skimage.transform import resize
from sklearn.decomposition import PCA
from perception import ColorImage, BinaryImage, GrayscaleImage
import json

def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, -min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=BLACK)

def normalize(color_im, crop_size=(512, 512)):

    # Center object in frame
    color_data = color_im.data
    nzp = color_im.nonzero_pixels().astype(np.int32)
    centroid = np.mean(nzp, axis=0)
    cx, cy = color_data.shape[1] // 2, color_data.shape[0] // 2
    color_data = imutils.translate(color_data, cx - round(centroid[1]), cy - round(centroid[0]))
    color_im = ColorImage(color_data, color_im.frame)

    # Crop about center to 512x512
    cx, cy = color_data.shape[1] // 2, color_data.shape[0] // 2
    crop_x = crop_size[0] / 2
    crop_y = crop_size[1] / 2
    color_data = imcrop(color_data, (cx-crop_x, cy-crop_y, cx+crop_x, cy+crop_y))
    color_im = ColorImage(color_data, color_im.frame)

    return color_im

# Limit number of CPU cores used.
cpu_cores = [8, 9, 10, 11] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

segmasks_dir = '/nfs/diskstation/dmwang/labeled_wisdom_real/phoxi/modal_segmasks'
images_dir = '/nfs/diskstation/dmwang/labeled_wisdom_real/phoxi/color_ims'

output_dir = '/nfs/diskstation/dmwang/labeled_wisdom_real/dataset'

# if os.path.exists(output_dir):
#     raise Exception("Output dataset directory already exists!")

# os.makedirs(output_dir)

for i in range(400):

    image_dir = os.path.join(output_dir, 'image_{:06d}'.format(i))

    os.makedirs(image_dir)

    image_filename = os.path.join(images_dir, 'image_{:06d}.png'.format(i))
    label_filename = os.path.join(images_dir, 'image_{:06d}.json'.format(i))
    mask_filename = os.path.join(segmasks_dir, 'image_{:06d}.png'.format(i))

    print('Processing Image: {}'.format(image_filename))

    image = ColorImage.open(image_filename)
    # print(image.data.shape)

    with open(label_filename) as f:
        label_data = json.load(f)['labels']
        # print(len(label_data['labels']))
        # for i in range(len(label_data['labels'])):
        #     print(label_data['labels'][i]['label_type'])
        #     print(label_data['labels'][i]['label_class'])
        #     print(label_data['labels'][i]['object_id'])

    masks = ColorImage.open(mask_filename).data
    # print(masks.shape)
    for j in range(len(label_data)):
        label_class = label_data[j]['label_class']
        if label_class:
            indices = np.where(masks == (j + 1))
            mask = np.zeros_like(masks)
            mask[indices] = 1.0
            modified_image = mask * image.data

            modified_image = ColorImage(modified_image)
            modified_image = normalize(modified_image)
            modified_image.save(os.path.join(image_dir, '{0}.png'.format(label_class)))
        # print(np.where(image.data == 5))
    # print(np.max(image.data), np.min(image.data))


    # print(image_filename, label_filename, mask_filename)
