"""
Class for interfacing with the Kinect v2 RGBD sensor
Author: Jeff Mahler
"""
import copy
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time

from core.constants import MM_TO_METERS, INTR_EXTENSION
import pylibfreenect2 as lf2

from camera_intrinsics import CameraIntrinsics
from image import ColorImage, DepthImage, IrImage, Image
from camera_sensor import CameraSensor

class Kinect2PacketPipelineMode:
    OPENGL = 0
    CPU = 1

class Kinect2FrameMode:
    COLOR_DEPTH = 0
    COLOR_DEPTH_IR = 1

class Kinect2RegistrationMode:
    NONE = 0
    COLOR_TO_DEPTH = 1

class Kinect2DepthMode:
    METERS = 0
    MILLIMETERS = 1

class Kinect2Sensor(CameraSensor):
    # constants for image height and width (in case they're needed somewhere)
    COLOR_IM_HEIGHT = 1080
    COLOR_IM_WIDTH = 1920
    DEPTH_IM_HEIGHT = 424
    DEPTH_IM_WIDTH = 512

    def __init__(self, packet_pipeline_mode = Kinect2PacketPipelineMode.CPU,
                 registration_mode = Kinect2RegistrationMode.COLOR_TO_DEPTH,
                 depth_mode = Kinect2DepthMode.METERS,
                 device_num=0, frame=None):
        """
        Initialize a Kinect v2 sensor with the given configuration
        Params:
           packet_pipeline_mode: OPENGL vs CPU packet processing
           registration_mode: mode of registering color image to IR camera frame of reference
           depth_mode: mode of storage for depth values in returned frame arrays
           device_num: number of the device on the USB bus
           frame: (str) name of the sensor's reference frame
        """
        self._device = None
        self._running = False
        self._packet_pipeline_mode = packet_pipeline_mode
        self._registration_mode = registration_mode
        self._depth_mode = depth_mode
        self._device_num = device_num
        self._frame = frame

        if self._frame is None:
            self._frame = 'kinect2_%d' %(self._device_num)
        self._color_frame = '%s_color' %(self._frame)
        self._ir_frame = self._frame # same as color since we normally use this one

    def __del__(self):
        """ Auto stop sensor for safety """
        if self.is_running:
            self.stop()

    @property
    def color_intrinsics(self):
        """ Color camera intrinsics """
        if self._device is None:
            raise RuntimeError('Kinect2 device %s not runnning. Cannot return color intrinsics')
        camera_params = self._device.getColorCameraParams()
        return CameraIntrinsics(self._color_frame, camera_params.fx, camera_params.fy,
                                camera_params.cx, camera_params.cy)

    @property
    def ir_intrinsics(self):
        """ Color camera intrinsics """
        if self._device is None:
            raise RuntimeError('Kinect2 device %s not runnning. Cannot return IR intrinsics')
        camera_params = self._device.getIrCameraParams()
        return CameraIntrinsics(self._ir_frame, camera_params.fx, camera_params.fy,
                                camera_params.cx, camera_params.cy,
                                height=Kinect2Sensor.DEPTH_IM_HEIGHT,
                                width=Kinect2Sensor.DEPTH_IM_WIDTH)

    @property
    def is_running(self):
        """ Whether or not the Kinect 2 stream is running """ 
        return self._running

    @property
    def frame(self):
        """ Reference frame of sensor """
        return self._frame

    @property
    def color_frame(self):
        """ Reference frame of color sensor """
        return self._color_frame

    @property
    def ir_frame(self):
        """ Reference frame of color sensor """
        return self._ir_frame

    def start(self):
        """ Starts the Kinect2 sensor stream """
        # open packet pipeline
        if self._packet_pipeline_mode == Kinect2PacketPipelineMode.OPENGL:
            self._pipeline = lf2.OpenGLPacketPipeline()
        elif self._packet_pipeline_mode == Kinect2PacketPipelineMode.CPU:
            self._pipeline = lf2.CpuPacketPipeline()
        
        # setup logger
        self._logger = lf2.createConsoleLogger(lf2.LoggerLevel.Warning)
        lf2.setGlobalLogger(self._logger)

        # check devices
        self._fn_handle = lf2.Freenect2()
        self._num_devices = self._fn_handle.enumerateDevices()
        if self._num_devices == 0:
            raise IOError('Failed to start stream. No Kinect2 devices available!')
        if self._num_devices <= self._device_num:
            raise IOError('Failed to start stream. Device num %d unavailable!' %(self._device_num))
            
        # open device
        self._serial = self._fn_handle.getDeviceSerialNumber(self._device_num)
        self._device = self._fn_handle.openDevice(self._serial, pipeline=self._pipeline)

        # add device sync modes
        self._listener = lf2.SyncMultiFrameListener(
            lf2.FrameType.Color | lf2.FrameType.Ir | lf2.FrameType.Depth)
        self._device.setColorFrameListener(self._listener)
        self._device.setIrAndDepthFrameListener(self._listener)
        
        # start device
        self._device.start()

        # open registration
        self._registration = None
        if self._registration_mode == Kinect2RegistrationMode.COLOR_TO_DEPTH:
            logging.debug('Using color to depth registration')
            self._registration = lf2.Registration(self._device.getIrCameraParams(),
                                                  self._device.getColorCameraParams())
        self._running = True

    def stop(self):
        """ Stops the Kinect2 sensor stream """
        # check that everything is running
        if not self._running or self._device is None:
            logging.warning('Kinect2 device %s not runnning. Aborting stop')
            return False

        # stop the device
        self._device.stop()
        self._device.close()
        self._device = None
        self._running = False
        return True

    def frames(self, skip_registration=False):
        """
        Return the color, depth, and ir frames
        Params:
           skip_registration: (bool) Optionally bypass registration
        """
        color_im, depth_im, ir_im, _ = self.frames_and_index_map(skip_registration=skip_registration)
        return color_im, depth_im, ir_im

    def frames_and_index_map(self, skip_registration=False):
        """
        Return the color, depth, ir frames, and color to depth index map 
        Params:
           skip_registration: (bool) Optionally bypass registration
        """
        if not self._running:
            raise RuntimeError('Kinect2 device %s not runnning. Cannot read frames' %(self._device_num))

        # read frames
        frames = self._listener.waitForNewFrame()
        unregistered_color = frames['color']
        distorted_depth = frames['depth']
        ir = frames['ir']

        # apply color to depth registration
        color_frame = self._color_frame
        color = unregistered_color
        depth = distorted_depth
        color_depth_map = np.zeros([depth.height, depth.width]).astype(np.int32).ravel()
        if not skip_registration and self._registration_mode == Kinect2RegistrationMode.COLOR_TO_DEPTH:
            color_frame = self._ir_frame
            depth = lf2.Frame(depth.width, depth.height, 4, lf2.FrameType.Depth)
            color = lf2.Frame(depth.width, depth.height, 4, lf2.FrameType.Color)
            self._registration.apply(unregistered_color, distorted_depth, depth, color, color_depth_map=color_depth_map)

        # convert to array (copy needed to prevent reference of deleted data
        color_arr = copy.copy(color.asarray())
        color_arr[:,:,[0,2]] = color_arr[:,:,[2,0]] # convert BGR to RGB
        color_arr[:,:,0] = np.fliplr(color_arr[:,:,0])
        color_arr[:,:,1] = np.fliplr(color_arr[:,:,1])
        color_arr[:,:,2] = np.fliplr(color_arr[:,:,2])
        color_arr[:,:,3] = np.fliplr(color_arr[:,:,3])
        depth_arr = np.fliplr(copy.copy(depth.asarray()))
        ir_arr = np.fliplr(copy.copy(ir.asarray()))

        # convert meters
        if self._depth_mode == Kinect2DepthMode.METERS:
            depth_arr = depth_arr * MM_TO_METERS

        # release and return
        self._listener.release(frames)
        return ColorImage(color_arr[:,:,:3], color_frame),\
            DepthImage(depth_arr, self._ir_frame), \
            IrImage(ir_arr.astype(np.uint16), self._ir_frame), \
            color_depth_map

    def get_median_depth_img(self, num_img=1):
        # get raw images
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        return Image.median_images(depths)

class VirtualKinect2Sensor(CameraSensor):
    """
    Class to spoof the Kinect2Sensor when using pre-captured test images
    """ 
    def __init__(self, path_to_images, frame=None):
        self._running = False
        self._path_to_images = path_to_images
        self._im_index = 0
        self._num_images = 0
        self._frame = frame
        filenames = os.listdir(self._path_to_images)

        # get number of images
        for filename in filenames:
            if filename.find('depth') != -1 and filename.endswith('.npy'):
                self._num_images += 1

        # set the frame dynamically
        if self._frame is None:
            for filename in filenames:
                file_root, file_ext = os.path.splitext(filename)
                color_ind = file_root.rfind('color')

                if file_ext == INTR_EXTENSION and color_ind != -1:
                    self._frame = file_root[:color_ind-1]
                    self._color_frame = file_root
                    self._ir_frame = file_root
                    break

        # load color intrinsics
        color_intr_filename = os.path.join(self._path_to_images, '%s_color.intr' %(self._frame))
        self._color_intr = CameraIntrinsics.load(color_intr_filename)
        ir_intr_filename = os.path.join(self._path_to_images, '%s_ir.intr' %(self._frame))
        self._ir_intr = CameraIntrinsics.load(ir_intr_filename)

    @property
    def path_to_images(self):
        return self._path_to_images

    @property
    def is_running(self):
        """ Whether or not the Kinect 2 stream is running """ 
        return self._running

    @property
    def frame(self):
        """ Reference frame of sensor """
        return self._frame

    @property
    def color_frame(self):
        """ Reference frame of color sensor """
        return self._color_frame

    @property
    def color_intrinsics(self):
        """ Color camera intrinsics """
        return self._color_intr

    @property
    def ir_intrinsics(self):
        """ IR camera intrinsics """
        return self._ir_intr

    @property
    def ir_frame(self):
        """ Reference frame of color sensor """
        return self._ir_frame

    def start(self):
        self._im_index = 0
        self._running = True

    def stop(self):
        self._running = False

    def frames(self):
        """
        Return the color, depth, and ir frames
        Params:
           skip_registration: (bool) Optionally bypass registration
        """
        if not self._running:
            raise RuntimeError('VirtualKinect2 device pointing to %s not runnning. Cannot read frames' %(self._path_to_images))
            
        if self._im_index > self._num_images:
            raise RuntimeError('VirtualKinect2 device is out of images')

        # read images
        color_filename = os.path.join(self._path_to_images, 'color_%d.png' %(self._im_index))
        color_im = ColorImage.open(color_filename, frame=self._frame)
        depth_filename = os.path.join(self._path_to_images, 'depth_%d.npy' %(self._im_index))
        depth_im = DepthImage.open(depth_filename, frame=self._frame)
        ir_filename = os.path.join(self._path_to_images, 'ir_%d.npy' %(self._im_index))
        ir_im = IrImage.open(ir_filename, frame=self._frame)
        self._im_index += 1
        return color_im, depth_im, ir_im

def load_images(cfg):
    """ Helper functions for loading a set of images """
    if 'prestored_data' in cfg.keys() and cfg['prestored_data'] == 1:
        sensor = VirtualKinect2Sensor(path_to_images=cfg['prestored_data_dir'], frame=cfg['sensor']['frame'])
    else:
        sensor = Kinect2Sensor(device_num=cfg['sensor']['device_num'], frame=cfg['sensor']['frame'],
                               packet_pipeline_mode=cfg['sensor']['pipeline_mode'])
    sensor.start()
    ir_intrinsics = sensor.ir_intrinsics

    # get raw images
    colors = []
    depths = []

    for _ in range(cfg['num_images']):
        color, depth, _ = sensor.frames()
        colors.append(color)
        depths.append(depth)

    sensor.stop()

    return colors, depths, ir_intrinsics

if __name__ == '__main__':
    # Simple test to read a single frame from the camera and display
    # NOTE: run examples/run_kinect2.py for a better demonstration of functionality
    logging.getLogger().setLevel(logging.DEBUG)

    sensor = Kinect2Sensor()
    sensor.start()

    frame_start = time.time()
    color, depth, ir = sensor.frames()
    frame_stop = time.time()
    logging.info('Frame read took %.4f sec' %(frame_stop - frame_start))

    color_intrinsics = sensor.color_intrinsics
    ir_intrinsics = sensor.ir_intrinsics
    logging.info('Color camera intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f' %(color_intrinsics.fx,
                                                                              color_intrinsics.fy,
                                                                              color_intrinsics.cx,
                                                                              color_intrinsics.cy))
    logging.info('IR camera intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f' %(ir_intrinsics.fx,
                                                                           ir_intrinsics.fy,
                                                                           ir_intrinsics.cx,
                                                                           ir_intrinsics.cy))

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(color)
    plt.title('Color', fontsize=15)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(depth, cmap=plt.cm.gray)
    plt.title('Depth', fontsize=15)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(ir, cmap=plt.cm.gray)
    plt.title('IR', fontsize=15)
    plt.axis('off')
    plt.show()

    sensor.stop()
