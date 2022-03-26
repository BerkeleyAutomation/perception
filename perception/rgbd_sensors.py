"""
RGBD Sensor factory
Author: Jeff Mahler
"""

from . import exceptions
from .virtual_camera_sensor import TensorDatasetVirtualSensor, VirtualSensor
from .webcam_sensor import WebcamSensor

try:
    from .kinect2_sensor import Kinect2Sensor
except BaseException as E:
    Kinect2Sensor = exceptions.closure(E)

try:
    from .kinect2_ros_sensor import KinectSensorBridged
except BaseException as E:
    KinectSensorBridged = exceptions.closure(E)

try:
    from .primesense_sensor import PrimesenseSensor
except BaseException as E:
    PrimesenseSensor = exceptions.closure(E)

try:
    from .primesense_ros_sensor import PrimesenseSensor_ROS
except BaseException as E:
    PrimesenseSensor_ROS = exceptions.closure(E)

try:
    from .realsense_sensor import RealSenseSensor
except BaseException as E:
    RealSenseSensor = exceptions.closure(E)

try:
    from .ensenso_sensor import EnsensoSensor
except BaseException as E:
    EnsensoSensor = exceptions.closure(E)

try:
    from .colorized_phoxi_sensor import ColorizedPhoXiSensor
    from .phoxi_ros_sensor import PhoXiSensor
except BaseException as E:
    PhoXiSensor = exceptions.closure(E)
    ColorizedPhoXiSensor = exceptions.closure(E)


class RgbdSensorFactory:
    """Factory class for Rgbd camera sensors."""

    @staticmethod
    def sensor(sensor_type, cfg):
        """Creates a camera sensor of the specified type.

        Parameters
        ----------
        sensor_type : :obj:`str`
            the type of the sensor (real or virtual)
        cfg : :obj:`YamlConfig`
            dictionary of parameters for sensor initialization
        """
        sensor_type = sensor_type.lower()
        if sensor_type == "kinect2":
            s = Kinect2Sensor(
                packet_pipeline_mode=cfg["pipeline_mode"],
                device_num=cfg["device_num"],
                frame=cfg["frame"],
            )
        elif sensor_type == "bridged_kinect2":
            s = KinectSensorBridged(quality=cfg["quality"], frame=cfg["frame"])
        elif sensor_type == "primesense":
            flip_images = True
            if "flip_images" in cfg.keys():
                flip_images = cfg["flip_images"]
            s = PrimesenseSensor(
                auto_white_balance=cfg["auto_white_balance"],
                flip_images=flip_images,
                frame=cfg["frame"],
            )
        elif sensor_type == "virtual":
            s = VirtualSensor(cfg["image_dir"], frame=cfg["frame"])
        elif sensor_type == "tensor_dataset":
            s = TensorDatasetVirtualSensor(
                cfg["dataset_dir"], frame=cfg["frame"]
            )
        elif sensor_type == "primesense_ros":
            s = PrimesenseSensor_ROS(frame=cfg["frame"])
        elif sensor_type == "ensenso":
            s = EnsensoSensor(frame=cfg["frame"])
        elif sensor_type == "phoxi":
            s = PhoXiSensor(
                frame=cfg["frame"],
                device_name=cfg["device_name"],
                size=cfg["size"],
            )
        elif sensor_type == "webcam":
            s = WebcamSensor(frame=cfg["frame"], device_id=cfg["device_id"])
        elif sensor_type == "colorized_phoxi":
            s = ColorizedPhoXiSensor(
                frame=cfg["frame"],
                phoxi_config=cfg["phoxi_config"],
                webcam_config=cfg["webcam_config"],
                calib_dir=cfg["calib_dir"],
            )
        elif sensor_type == "realsense":
            s = RealSenseSensor(
                cam_id=cfg["cam_id"],
                filter_depth=cfg["filter_depth"],
                frame=cfg["frame"],
            )
        else:
            raise ValueError(
                "RGBD sensor type %s not supported" % (sensor_type)
            )
        return s
