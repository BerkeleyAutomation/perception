"""
Classes for features of a 3D object surface.
Author: Jeff
"""
from abc import ABCMeta, abstractmethod

import numpy as np

class Feature:
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

class LocalFeature(Feature):
    """ Local (e.g. pointwise) features on shape surfaces """
    __metaclass__ = ABCMeta

    def __init__(self, descriptor, rf, point, normal):
        """
        Params:
           descriptor: vector to describe the point
           rf: reference frame of the descriptor
           point: 3D point on shape surface that descriptor corresponds to
           normal: 3D surface normal on shape surface at corresponding point
        """
        self.descriptor_ = descriptor
        self.rf_ = rf
        self.point_ = point
        self.normal_ = normal

    @property
    def descriptor(self):
        return self.descriptor_

    @property
    def reference_frame(self):
        return self.rf_

    @property
    def keypoint(self):
        return self.point_

    @property
    def normal(self):
        return self.normal_

class GlobalFeature(Feature):
    """ Global features of a full shape surface """
    __metaclass__ = ABCMeta

    def __init__(self, key, descriptor, pose):
        """
        Params:
           key: object key in database that descriptor corresponds to
           descriptor: vector to describe the object
           pose: pose of object for the descriptor
        """
        self.key_ = key
        self.descriptor_ = descriptor
        self.pose_ = pose

    @property
    def key(self):
        return self.key_

    @property
    def descriptor(self):
        return self.descriptor_

    @property
    def pose(self):
        return self.pose_

class SHOTFeature(LocalFeature):
    """ Same interface as standard local feature """ 
    def __init__(self, descriptor, rf, point, normal):
        LocalFeature.__init__(self, descriptor, rf, point, normal)

class MVCNNFeature(GlobalFeature):
    """ Same interface as standard global feature """ 
    def __init__(self, key, descriptor, pose=None):
        GlobalFeature.__init__(self, key, descriptor, pose)

class BagOfFeatures:
    """ Actually just a list of features, but created for the sake of future bag-of-words reps """
    def __init__(self, features = None):
        self.features_ = features
        if self.features_ is None:
            self.features_ = []

        self.num_features_ = len(self.features_)

    def add(self, feature):
        """ Add a new feature to the bag """
        self.features_.append(feature)
        self.num_features_ = len(self.features_)        

    def extend(self, features):
        """ Add a list of features to the bag """
        self.features_.extend(features)
        self.num_features_ = len(self.features_)        

    def feature(self, index):
        """ Returns a feature """
        if index < 0 or index >= self.num_features_:
            raise ValueError('Index %d out of range' %(index))
        return self.features_[index]

    def feature_subset(self, indices):
        """ Returns some subset of the features """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if not isinstance(indices, list):
            raise ValueError('Can only index with lists')
        return [self.features_[i] for i in indices]

    def to_hdf5(self, h):
        """ Convert to hdf5 object """
        h.create_dataset('descriptors', data=self.descriptors)
        h.create_dataset('reference_frames', data=self.reference_frames)
        h.create_dataset('keypoints', data=self.keypoints)
        h.create_dataset('normals', data=self.normals)

    @property
    def num_features(self):
        return self.num_features_

    @property
    def descriptors(self):
        """ Make a nice array of the descriptors """
        return np.array([f.descriptor for f in self.features_])

    @property
    def reference_frames(self):
        """ Make a nice array of the reference frames """
        return np.array([f.reference_frame for f in self.features_])

    @property
    def keypoints(self):
        """ Make a nice array of the keypoints """
        return np.array([f.keypoint for f in self.features_])

    @property
    def normals(self):
        """ Make a nice array of the normals """
        return np.array([f.normal for f in self.features_])
