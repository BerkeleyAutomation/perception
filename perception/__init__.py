'''
Alan Perception Module.
Authors: Jeff, Jacky
'''
import logging
from camera_intrinsics import CameraIntrinsics
try:
    from cnn import AlexNet, AlexNetWeights
    from feature_extractors import CNNBatchFeatureExtractor
except Exception, e:
    logging.warn("Cannot import CNNBatchFeatureExtractor: \n{0}".format(str(e)))
from feature_matcher import FeatureMatcher, PointToPlaneFeatureMatcher
from features import LocalFeature, GlobalFeature, SHOTFeature, MVCNNFeature, BagOfFeatures
from image import Image, ColorImage, DepthImage, IrImage, GrayscaleImage, BinaryImage
try:
    from kinect2_sensor import Kinect2PacketPipelineMode, Kinect2FrameMode, Kinect2RegistrationMode, Kinect2DepthMode, Kinect2Sensor, VirtualKinect2Sensor, load_images
except Exception:
    logging.warn("Cannot import kinect2_sensor!")
from opencv_camera_sensor import OpenCVCameraSensor
from registration import IterativeRegistrationSolver, PointToPlaneICPSolver, RegistrationResult
from video_recorder import VideoRecorder
from video_writer import write_video


__all__ = ['Image', 'ColorImage', 'DepthImage', 'IrImage', 'GrayscaleImage',
          'AlexNet', 'AlexNetWeights',
          'Kinect2PacketPipelineMode', 'Kinect2FrameMode', 'Kinect2RegistrationMode','Kinect2DepthMode', 'Kinect2Sensor',
          'CameraIntrinsics',
          'CNNBatchFeatureExtractor',
          'FeatureMatcher','PointToPlaneFeatureMatcher',
          'IterativeRegistrationSolver','PointToPlaneICPSolver','RegistrationResult',
          "LocalFeature", "GlobalFeature", "SHOTFeature", "MVCNNFeature", "BagOfFeatures",
          'VideoRecorder',
          'OpenCVCameraSensor',
]
