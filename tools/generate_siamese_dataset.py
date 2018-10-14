import numpy as np
import imutils
import os
import cv2
import random
import uuid
from skimage.transform import resize
from sklearn.decomposition import PCA
from perception import ColorImage, BinaryImage

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

def normalize_fill(color_im, crop_size=(512, 512)):

    # Crop image to bounds of object
    nzp = color_im.nonzero_pixels()
    ymin, xmin = np.min(nzp, axis=0)
    ymax, xmax = np.max(nzp, axis=0)
    color_data = color_im.data[ymin:ymax, xmin:xmax, :]

    # Resize to square by padding out the smaller dimension
    xlen, ylen = color_data.shape[1], color_data.shape[0]
    padding = xlen - ylen
    padding_l = abs(padding) / 2
    padding_r = abs(padding) - padding_l
    if padding > 0:
        color_data = np.pad(color_data,
                            ((padding_l, padding_r), (0,0), (0,0)),
                            mode='constant')
    elif padding < 0:
        color_data = np.pad(color_data,
                            ((0,0), (padding_l, padding_r), (0,0)),
                            mode='constant')

    color_data = resize(color_data, (crop_size[1], crop_size[0], 3), clip=False, preserve_range=True).astype(np.uint8)

    return ColorImage(color_data, color_im.frame)

def get_key_function(central_point):

    def key_function(point):
        vec = point - central_point
        angle = np.arctan2(vec[0], vec[1])
        return angle

    return key_function

def augment(image, n_samples, crop_size, preserve_scale):
    """Create data augmentations of an image crop by randomly occluding it with a line.
    """
    samples = []
    if preserve_scale:
        samples.append(normalize(image))
    else:
        samples.append(normalize_fill(image, crop_size=crop_size))
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
            if preserve_scale:
                img = normalize(image.mask_binary(mask), crop_size=crop_size)
            else:
                img = normalize_fill(image.mask_binary(mask), crop_size=crop_size)
            samples.append(img)

        if np.count_nonzero(upper_mask) > min_number_points:
            mask = BinaryImage(upper_mask)
            if preserve_scale:
                img = normalize(image.mask_binary(mask), crop_size=crop_size)
            else:
                img = normalize_fill(image.mask_binary(mask), crop_size=crop_size)
            samples.append(img)

    return samples[:n_samples]


if __name__ == '__main__':
    object_images_dir = '/nfs/diskstation/projects/mech_search/siamese_net_training/single_obj_dataset/phoxi/color_images'
    output_dataset_dir = '/nfs/diskstation/dmwang/mech_search_data3'
    object_train_split = 0.8
    num_images_per_view = 10
    preserve_scale = True
    crop_size = (512, 512)

    if os.path.exists(output_dataset_dir):
        raise Exception("Output dataset directory already exists!")

    train_dir = os.path.join(output_dataset_dir, 'train')
    validation_dir = os.path.join(output_dataset_dir, 'validation')
    orig_dir = os.path.join(output_dataset_dir, 'originals')

    os.makedirs(output_dataset_dir)
    os.makedirs(orig_dir)
    os.makedirs(train_dir)
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

    # Split set of objects into training and validation.
    split_index = int(object_train_split * len(object_images))
    all_objects = object_images.keys()
    random.shuffle(all_objects)
    train_objects = all_objects[:split_index]
    validation_objects = all_objects[split_index:]

    for objects, directory in [(train_objects, train_dir), (validation_objects, validation_dir)]:
        for objname in objects:
            output_dir = os.path.join(directory, objname)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            image_names = object_images[objname]
            for i, fn in enumerate(image_names):
                print(fn)
                path, base = os.path.split(fn)
                image = ColorImage.open(fn)
                samples = augment(image, num_images_per_view, crop_size, preserve_scale)

                # Save original, which is always first sample
                orig_output_dir = os.path.join(orig_dir, objname)
                if not os.path.exists(orig_output_dir):
                    os.makedirs(orig_output_dir)
                orig = samples[0]
                orig.save(os.path.join(orig_output_dir, 'view_{:06d}.png'.format(i)))

                # Save samples
                samples_output_dir = os.path.join(output_dir, 'view_{:06d}'.format(i))
                if not os.path.exists(samples_output_dir):
                    os.makedirs(samples_output_dir)
                for sample in samples:
                    sample_name = uuid.uuid4().hex
                    sample.save(os.path.join(samples_output_dir, '{}.png'.format(sample_name)))
