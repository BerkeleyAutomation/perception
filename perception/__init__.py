'''
Alan Perception Module.
Authors: Jeff, Jacky
'''
from camera_intrinsics import CameraIntrinsics
try:
    from cnn import AlexNet, AlexNetWeights, conv
    from feature_extractors import FeatureExtractor, CNNBatchFeatureExtractor, CNNReusableBatchFeatureExtractor
except Exception:
    print 'Unable to import ConvNet modules! Likely due to missing tensorflow.'
    print 'TensorFlow can be installed following the instructions in https://www.tensorflow.org/get_started/os_setup'
from feature_matcher import Correspondences, NormalCorrespondences, FeatureMatcher, RawDistanceFeatureMatcher, PointToPlaneFeatureMatcher
from features import Feature, LocalFeature, GlobalFeature, SHOTFeature, MVCNNFeature, BagOfFeatures
from image import Image, ColorImage, DepthImage, IrImage, GrayscaleImage, RgbdImage, GdImage, SegmentationImage, BinaryImage, PointCloudImage, NormalCloudImage
from object_render import RenderMode, ObjectRender, QueryImageBundle
from chessboard_registration import ChessboardRegistrationResult, CameraChessboardRegistration
from point_registration import RegistrationResult, IterativeRegistrationSolver, PointToPlaneICPSolver
from detector import RgbdDetection, RgbdDetector, RgbdForegroundMaskDetector, RgbdForegroundMaskQueryImageDetector, PointCloudBoxDetector, RgbdDetectorFactory

from camera_sensor import CameraSensor

try:
    from cnn import AlexNetWeights, AlexNet, conv
except Exception:
    print 'Unable to import ConvNet modules! Likely due to missing tensorflow.'
    print 'TensorFlow can be installed following the instructions in https://www.tensorflow.org/get_started/os_setup'
    
try:
    from kinect2_sensor import Kinect2PacketPipelineMode, Kinect2FrameMode, Kinect2RegistrationMode, Kinect2DepthMode, Kinect2Sensor, VirtualKinect2Sensor, Kinect2SensorFactory, load_images
except Exception:
    print 'Unable to import Kinect2 sensor modules! Likely due to missing pylibfreenect2.'
    print 'The pylibfreenect2 library can be installed from https://github.com/r9y9/pylibfreenect2'

try:
    from primesense_sensor import PrimesenseSensor, VirtualPrimesenseSensor, PrimesenseSensor_ROS, PrimesenseRegistrationMode
except Exception:
    print 'Unable to import Kinect2 sensor modules! Likely due to missing pylibfreenect2.'
    print 'The pylibfreenect2 library can be installed from https://github.com/r9y9/pylibfreenect2'

from opencv_camera_sensor import OpenCVCameraSensor
from rgbd_sensors import RgbdSensorFactory
from video_recorder import VideoRecorder
from video_writer import write_video

__all__ = [
    'CameraIntrinsics',
    'AlexNetWeights', 'AlexNet', 'conv',
    'RgbdDetection', 'RgbdDetector', 'RgbdForegroundMaskDetector', 'RgbdForegroundMaskQueryImageDetector', 'PointCloudBoxDetector', 'RgbdDetectorFactory',
    'FeatureExtractor', 'CNNBatchFeatureExtractor', 'CNNReusableBatchFeatureExtractor',
    'Correspondences', 'NormalCorrespondences', 'FeatureMatcher', 'RawDistanceFeatureMatcher', 'PointToPlaneFeatureMatcher',
    'Feature', 'LocalFeature', 'GlobalFeature', 'SHOTFeature', 'MVCNNFeature', 'BagOfFeatures',
    'Image', 'ColorImage', 'DepthImage', 'IrImage', 'GrayscaleImage', 'RgbdImage', 'GdImage', 'SegmentationImage', 'BinaryImage', 'PointCloudImage', 'NormalCloudImage',
    'Kinect2PacketPipelineMode', 'Kinect2FrameMode', 'Kinect2RegistrationMode', 'Kinect2DepthMode', 'Kinect2Sensor', 'VirtualKinect2Sensor', 'Kinect2SensorFactory', 'load_images',
    'RgbdSensorFactory', 'PrimesenseSensor', 'VirtualPrimesenseSensor', 'PrimesenseSensor_ROS', 'PrimesenseRegistrationMode',
    'RenderMode', 'ObjectRender', 'QueryImageBundle',
    'RegistrationResult', 'IterativeRegistrationSolver', 'PointToPlaneICPSolver',
    'OpenCVCameraSensor',
    'VideoRecorder',
]
