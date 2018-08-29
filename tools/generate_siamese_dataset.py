import numpy as np
import imutils
import os
import cv2
import random
import uuid
from skimage.transform import resize
from sklearn.decomposition import PCA
from perception import ColorImage, BinaryImage

from visualization import Visualizer2D as vis2d

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

    # Rotate via PCA so that the principal axis is vertical
    color_data = color_im.data
    nzp = color_im.nonzero_pixels().astype(np.int32)
    nzp = nzp - np.mean(nzp, axis=0)
    pca = PCA(n_components=2)
    pca.fit(nzp)
    axis = pca.components_[0]
    if axis[0] != 0:
        angle = np.rad2deg(np.arctan(axis[1]/axis[0]))
    else:
        angle = 90.0
    color_data = imutils.rotate_bound(color_data, angle)
    cx, cy = color_data.shape[1] // 2, color_data.shape[0] // 2

    # Crop about center to 512x512
    crop_x = crop_size[0] / 2
    crop_y = crop_size[1] / 2
    color_data = imcrop(color_data, (cx-crop_x, cy-crop_y, cx+crop_x, cy+crop_y))
    color_im = ColorImage(color_data, color_im.frame)

    return color_im

def get_key_function(central_point):

    def key_function(point):
        vec = point - central_point
        angle = np.arctan2(vec[0], vec[1])
        return angle

    return key_function

def augment(image, n_samples):
    """Create data augmentations of an image crop by randomly occluding it with a line.
    """

    samples = [normalize(image)]
    orig_mask = image.to_binary()
    nzp = orig_mask.nonzero_pixels()
    min_number_points = 0.35*np.count_nonzero(orig_mask.data)

    while len(samples) < n_samples:
        # Sample point in image
        idx = np.random.randint(len(nzp))
        y, x = nzp[idx]
        slope = np.tan(np.random.uniform(0, np.pi))
        intercept = -y - slope * x

        # Find points of intersection with box
        max_y = orig_mask.shape[0] - 1
        max_x = orig_mask.shape[1] - 1
        inter_points = []
        left_y = -intercept
        right_y = -slope * max_x - intercept
        top_x = -intercept / slope
        bot_x = -(max_y + intercept) / slope

        if left_y > 0 and left_y < max_y:
            inter_points.append([0, int(left_y)])
        if top_x > 0 and top_x < max_x:
            inter_points.append([int(top_x), 0])
        if right_y > 0 and right_y < max_y:
            inter_points.append([max_x, int(right_y)])
        if bot_x > 0 and bot_x < max_x:
            inter_points.append([int(bot_x), max_y])

        inter_points = np.array(inter_points, dtype=np.int)
        if len(inter_points) != 2:
            print("ERROR")
            continue

        # Partition corner points based on above/below line
        corner_points = np.array([[0,0],[0,max_y],[max_x, max_y],[max_x,0]])
        lower_points = []
        upper_points = []
        for point in corner_points:
            if point[1] > -(slope * point[0] + intercept):
                upper_points.append(point)
            else:
                lower_points.append(point)

        lower_points = np.array(lower_points)
        upper_points = np.array(upper_points)

        # Create polygons
        try:
            lower_polygon = np.vstack((inter_points, lower_points))
            upper_polygon = np.vstack((inter_points, upper_points))
        except:
            import pdb
            pdb.set_trace()
        lower_polygon = np.array(sorted(lower_polygon, key=get_key_function(np.mean(lower_polygon, axis=0))))
        upper_polygon = np.array(sorted(upper_polygon, key=get_key_function(np.mean(upper_polygon, axis=0))))

        # Create masks
        lower_mask = orig_mask.data.copy()
        upper_mask = orig_mask.data.copy()
        cv2.fillConvexPoly(lower_mask, lower_polygon, 0)
        cv2.fillConvexPoly(upper_mask, upper_polygon, 0)

        if np.count_nonzero(lower_mask) > min_number_points:
            mask = BinaryImage(lower_mask)
            img = normalize(image.mask_binary(mask))
            samples.append(img)

        if np.count_nonzero(upper_mask) > min_number_points:
            mask = BinaryImage(upper_mask)
            img = normalize(image.mask_binary(mask))
            samples.append(img)

    return samples[:n_samples]


if __name__ == '__main__':
    object_images_dir = '/nfs/diskstation/projects/mech_search/siamese_net_training/single_obj_dataset/phoxi/color_images'
    output_dataset_dir = '/nfs/diskstation/projects/mech_search/siamese_net_training/phoxi_training_dataset'
    per_obj_train_split = 0.8

    train_dir = os.path.join(output_dataset_dir, 'train')
    validation_dir = os.path.join(output_dataset_dir, 'validation')

    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    # Generate training and validation data

    # Collect per-object lists of files
    object_images = {}
    for filename in os.listdir(object_images_dir):
        # Extract object that this image corresponds to
        base, _ = os.path.splitext(filename)
        objname = base.rsplit('_', 1)[0]

        if objname not in object_images:
            object_images[objname] = []

        object_images[objname].append(os.path.join(object_images_dir, filename))

    # Split them into validation and training
    for objname in object_images:
        train_output_dir = os.path.join(train_dir, objname)
        validation_output_dir = os.path.join(validation_dir, objname)
        if not os.path.exists(train_output_dir):
            os.makedirs(train_output_dir)
        if not os.path.exists(validation_output_dir):
            os.makedirs(validation_output_dir)

        image_names = object_images[objname]
        samples = []
        for fn in image_names:
            path, base = os.path.split(fn)
            image = ColorImage.open(fn)
            samples.extend(augment(image, 10))

        random.shuffle(samples)
        cutoff = int(per_obj_train_split * len(samples))

        for i, sample in enumerate(samples):
            sample_name = uuid.uuid4().hex
            output_dir = train_output_dir
            if i >= cutoff:
                output_dir = validation_output_dir
            sample.save(os.path.join(output_dir, '{}.png'.format(sample_name)))
