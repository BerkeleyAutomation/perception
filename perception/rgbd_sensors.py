"""
RGBD Sensor factory
Author: Jeff Mahler
"""
from perception import Kinect2Sensor, VirtualKinect2Sensor, PrimesenseSensor, VirtualPrimesenseSensor, PrimesenseSensor_ROS

class RgbdSensorFactory:
    """ Factory class for Rgbd camera sensors. """

    @staticmethod
    def sensor(sensor_type, cfg):
        """ Creates a camera sensor of the specified type.

        Parameters
        ----------
        sensor_type : :obj:`str`
            the type of the sensor (real or virtual)
        cfg : :obj:`YamlConfig`
            dictionary of parameters for sensor initialization
        """
        sensor_type = sensor_type.lower()
        if sensor_type == 'kinect2':
            s = Kinect2Sensor(packet_pipeline_mode=cfg['pipeline_mode'],
                              device_num=cfg['device_num'],
                              frame=cfg['frame'])
        elif sensor_type == 'virtual_kinect2':
            s = VirtualKinect2Sensor(cfg['image_dir'],
                                     frame=cfg['frame'])
        elif sensor_type == 'primesense':
            flip_images = True
            if 'flip_images' in cfg.keys():
                flip_images = cfg['flip_images']
            s = PrimesenseSensor(auto_white_balance=cfg['auto_white_balance'],
                                 flip_images=flip_images,
                                 frame=cfg['frame'])
        elif sensor_type == 'virtual_primesense':
            s = VirtualPrimesenseSensor(cfg['image_dir'],
                                        frame=cfg['frame'])
        elif sensor_type == 'primesense_ros':
            s = PrimesenseSensor_ROS(frame=cfg['frame'])
        else:
            raise ValueError('RGBD sensor type %s not supported' %(sensor_type))
        return s
