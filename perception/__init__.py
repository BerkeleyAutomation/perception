"""
Alan Perception Module.
Authors: Jeff, Jacky
"""
import logging

from .camera_intrinsics import CameraIntrinsics

try:
    from .features import (
        Feature,
        LocalFeature,
        GlobalFeature,
        SHOTFeature,
        MVCNNFeature,
        BagOfFeatures,
    )
except ImportError:
    logging.warning(
        "Unable to import CNN modules! Likely due to missing tensorflow."
    )
    logging.warning(
        "TensorFlow can be installed following the instructions "
        "in https://www.tensorflow.org/get_started/os_setup"
    )

from .feature_matcher import (
    Correspondences,
    NormalCorrespondences,
    FeatureMatcher,
    RawDistanceFeatureMatcher,
    PointToPlaneFeatureMatcher,
)
from .image import (
    Image,
    ColorImage,
    DepthImage,
    IrImage,
    GrayscaleImage,
    RgbdImage,
    GdImage,
    SegmentationImage,
    BinaryImage,
    PointCloudImage,
    NormalCloudImage,
)
from .chessboard_registration import (
    ChessboardRegistrationResult,
    CameraChessboardRegistration,
)
from .point_registration import (
    RegistrationResult,
    IterativeRegistrationSolver,
    PointToPlaneICPSolver,
)
from .detector import (
    RgbdDetection,
    RgbdDetector,
    RgbdForegroundMaskDetector,
    RgbdForegroundMaskQueryImageDetector,
    PointCloudBoxDetector,
    RgbdDetectorFactory,
)
from .camera_sensor import CameraSensor
from .virtual_camera_sensor import (
    VirtualSensor,
    TensorDatasetVirtualSensor,
)
from .webcam_sensor import WebcamSensor

try:
    from .kinect2_sensor import (
        Kinect2PacketPipelineMode,
        Kinect2FrameMode,
        Kinect2RegistrationMode,
        Kinect2DepthMode,
        Kinect2BridgedQuality,
        Kinect2Sensor,
        KinectSensorBridged,
        VirtualKinect2Sensor,
        Kinect2SensorFactory,
        load_images,
    )
except ImportError:
    logging.warning(
        "Unable to import Kinect2 sensor modules! "
        "Likely due to missing pylibfreenect2."
    )
    logging.warning(
        "The pylibfreenect2 library can be installed "
        "from https://github.com/r9y9/pylibfreenect2"
    )

try:
    from .primesense_sensor import (
        PrimesenseSensor,
        PrimesenseRegistrationMode,
    )
except ImportError:
    logging.warning(
        "Unable to import Primsense sensor modules! "
        "Likely due to missing OpenNI2."
    )
try:
    from .primesense_ros_sensor import PrimesenseSensor_ROS
except ImportError:
    logging.warning("Unable to import Primesense ROS module!")

try:
    from .realsense_sensor import RealSenseSensor
except ImportError:
    logging.warning("Unable to import RealSense sensor modules!")

try:
    from .ensenso_sensor import EnsensoSensor
except ImportError:
    logging.warning("Unable to import Ensenso sensor modules!.")

try:
    from .phoxi_sensor import PhoXiSensor
    from .colorized_phoxi_sensor import ColorizedPhoXiSensor
except ImportError:
    logging.warning("Unable to import PhoXi sensor modules!")

try:
    from .opencv_camera_sensor import OpenCVCameraSensor
    from .rgbd_sensors import RgbdSensorFactory
except ImportError:
    logging.warning("Unable to import generic sensor modules!.")

try:
    from .weight_sensor import WeightSensor
except ImportError:
    logging.warning("Unable to import weight sensor modules!")

from .video_recorder import VideoRecorder
