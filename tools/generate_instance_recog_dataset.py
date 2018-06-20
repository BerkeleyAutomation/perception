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

class Cluster(object):
    """A cluster of pixels in an image that represents an object hypothesis.
    """
    def __init__(self, mask, color_im, crop_size=(224, 224)):
        """Create an object cluster.

        Parameters
        ----------
        mask : autolab_perception.BinaryImage
            A mask for the cluster in the target image.
        color_im : autolab_perception.ColorImage
            The target color image.
        model : keras model
            A Keras model for featurizing image crops.
        crop_size : (2,) int
            The size of crops to be fed to the model in pixels.
        """
        self._mask = mask
        self._fullsize_mask = None
        self._valid = True

        # Create image masked crop
        xinds, yinds = np.where(mask.data)
        xmin, xmax = np.min(xinds), np.max(xinds) + 1
        ymin, ymax = np.min(yinds), np.max(yinds) + 1
        color_data = np.zeros(color_im.data.shape).astype(np.uint8)
        color_data[xinds, yinds, :] = color_im.data[xinds, yinds, :]
        color_data = color_data[xmin:xmax, ymin:ymax, :]

        # PCA + rotation
        nzp = mask.nonzero_pixels().astype(np.int32)
        nzp = nzp - np.mean(nzp, axis=0)
        pca = PCA(n_components=2)
        pca.fit(nzp)
        axis = pca.components_[0]
        if axis[0] != 0:
            angle = np.rad2deg(np.arctan(axis[1]/axis[0]))
        else:
            angle = 90.0
        color_data = imutils.rotate_bound(color_data, angle)

        # Save bounds, bb center, and mass
        self._bounds = np.array([[xmin, ymin], [xmax, ymax]])
        self._bb_center = np.mean(self._bounds, axis=0)
        self._mass = len(xinds)

        # Resize to square
        xlen, ylen = color_data.shape[0], color_data.shape[1]
        padding = xlen - ylen
        padding_l = abs(padding) / 2
        padding_r = abs(padding) - padding_l
        if padding > 0:
            color_data = np.pad(color_data,
                                ((0,0), (padding_l, padding_r), (0,0)),
                                mode='constant')
        elif padding < 0:
            color_data = np.pad(color_data,
                                ((padding_l, padding_r), (0,0), (0,0)),
                                mode='constant')


        # Resize to crop size
        color_data = resize(color_data, (crop_size[0], crop_size[1], 3),
                            clip=False, preserve_range=True).astype(np.uint8)

        # Save crop and features
        self._crop = ColorImage(color_data)

    @property
    def mask(self):
        """autolab_perception.BinaryImage : A mask for the cluster in the target image.
        """
        return self._mask


    @property
    def valid(self):
        """bool : False if the cluster is known to have been picked out
        of the scene.
        """
        return self._valid


    @valid.setter
    def valid(self, v):
        self._valid = bool(v)


    @property
    def crop(self):
        """autolab_perception.ColorImage : The cropped color image
        that was used for featurization.
        """
        return self._crop


    @property
    def bounds(self):
        """(2,2) float : The lower and upper bounds of the cluster in
        pixel coordinates.
        """
        return self._bounds


    @property
    def bb_center(self):
        """(2,) float : The center of the cluster's bounding box in pixel coordinates.
        """
        return self._bb_center


    @property
    def mass(self):
        """int : The number of pixels in the cluster.
        """
        return self._mass

def get_key_function(central_point):

    def key_function(point):
        vec = point - central_point
        angle = np.arctan2(vec[0], vec[1])
        return angle

    return key_function

def augment(image, n_samples):
    """Create data augmentations of an image crop by randomly occluding it with a line.
    """
    samples = []
    orig_mask = image.to_binary()
    min_number_points = 0.35*np.count_nonzero(orig_mask.data)
    while len(samples) < n_samples:
        # Sample point in image
        x = np.random.randint(image.shape[1])
        y = np.random.randint(image.shape[0])
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
            print("SHIT")
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
            c = Cluster(BinaryImage(lower_mask), cluster.crop)
            samples.append(c.crop)

        if np.count_nonzero(upper_mask) > min_number_points:
            c = Cluster(BinaryImage(upper_mask), cluster.crop)
            if len(samples) < n_samples:
                samples.append(c.crop)

    #for sample in samples:
    #    vis2d.figure()
    #    vis2d.imshow(sample)
    #    vis2d.show()
    return samples



if __name__ == '__main__':
    object_images_dir = '/nfs/diskstation/projects/dex-net/segmentation/physical_experiments/single_obj_images/phoxi/images'
    output_dataset_dir = '/nfs/diskstation/projects/dex-net/segmentation/physical_experiments/single_obj_dataset'
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
        random.shuffle(image_names)
        cutoff = int(per_obj_train_split * len(image_names))
        train_filenames = image_names[:cutoff]
        validation_filenames = image_names[cutoff:]

        for fn in train_filenames:
            path, base = os.path.split(fn)
            image = ColorImage.open(fn)
            cluster = Cluster(image.to_binary(), image)
            samples = augment(cluster.crop, 10)
            for sample in samples:
                sample_name = uuid.uuid4().hex
                sample.save(os.path.join(train_output_dir, '{}.png'.format(sample_name)))

        for fn in validation_filenames:
            path, base = os.path.split(fn)
            image = ColorImage.open(fn)
            cluster = Cluster(image.to_binary(), image)
            samples = augment(cluster.crop, 10)
            for sample in samples:
                sample_name = uuid.uuid4().hex
                sample.save(os.path.join(validation_output_dir, '{}.png'.format(sample_name)))
