"""
Class for interfacing with the Kinect v2 RGBD sensor
Author: Jeff Mahler
"""
import copy
import logging
import numpy as np
import os

from constants import MM_TO_METERS, INTR_EXTENSION
try:
    from primesense import openni2
except:
    logging.warning('Unable to import openni2 driver. Python-only Primesense driver may not work properly')

from perception import CameraIntrinsics, CameraSensor, ColorImage, DepthImage, IrImage, Image

try:
    import rospy
    from rospy import numpy_msg
except ImportError:
    logging.warning("Failed to import ROS in primesense_sensor.py. ROS functionality not available")
try:
    from perception.srv import *
    ImageBufferResponse = rospy.numpy_msg.numpy_msg(ImageBufferResponse)
    ImageBuffer._response_class = ImageBufferResponse
except ImportError:
    logging.warning('primesense_sensor.py not installed as catkin package. ROS functionality not available.')

class PrimesenseRegistrationMode:
    """Primesense registration mode.
    """
    NONE = 0
    DEPTH_TO_COLOR = 1

class PrimesenseSensor(CameraSensor):
    """ Class for interacting with a Primesense RGBD sensor.
    """
    #Constants for image height and width (in case they're needed somewhere)
    COLOR_IM_HEIGHT = 480
    COLOR_IM_WIDTH = 640
    DEPTH_IM_HEIGHT = 480
    DEPTH_IM_WIDTH = 640
    CENTER_X = float(DEPTH_IM_WIDTH-1) / 2.0
    CENTER_Y = float(DEPTH_IM_HEIGHT-1) / 2.0
    FOCAL_X = 525.
    FOCAL_Y = 525.
    FPS = 30
    OPENNI2_PATH = '/home/autolab/Libraries/OpenNI-Linux-x64-2.2/Redist'

    def __init__(self, registration_mode=PrimesenseRegistrationMode.DEPTH_TO_COLOR,
                 auto_white_balance=False, auto_exposure=True,
                 enable_depth_color_sync=True, flip_images=True, frame=None):
        self._device = None
        self._depth_stream = None
        self._color_stream = None
        self._running = None

        self._registration_mode = registration_mode
        self._auto_white_balance = auto_white_balance
        self._auto_exposure = auto_exposure
        self._enable_depth_color_sync = enable_depth_color_sync
        self._flip_images = flip_images

        self._frame = frame

        if self._frame is None:
            self._frame = 'primesense'
        self._color_frame = '%s_color' %(self._frame)
        self._ir_frame = self._frame # same as color since we normally use this one

    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()

    @property
    def color_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the primesense color camera.
        """
        return CameraIntrinsics(self._ir_frame, PrimesenseSensor.FOCAL_X, PrimesenseSensor.FOCAL_Y,
                                PrimesenseSensor.CENTER_X, PrimesenseSensor.CENTER_Y,
                                height=PrimesenseSensor.DEPTH_IM_HEIGHT,
                                width=PrimesenseSensor.DEPTH_IM_WIDTH)

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the primesense IR camera.
        """
        return CameraIntrinsics(self._ir_frame, PrimesenseSensor.FOCAL_X, PrimesenseSensor.FOCAL_Y,
                                PrimesenseSensor.CENTER_X, PrimesenseSensor.CENTER_Y,
                                height=PrimesenseSensor.DEPTH_IM_HEIGHT,
                                width=PrimesenseSensor.DEPTH_IM_WIDTH)

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise.
        """
        return self._running

    @property
    def frame(self):
        """:obj:`str` : The reference frame of the sensor.
        """
        return self._frame

    @property
    def color_frame(self):
        """:obj:`str` : The reference frame of the color sensor.
        """
        return self._color_frame

    @property
    def ir_frame(self):
        """:obj:`str` : The reference frame of the IR sensor.
        """
        return self._ir_frame
    
    def start(self):
        """ Start the sensor """
        # open device
        openni2.initialize(PrimesenseSensor.OPENNI2_PATH)
        self._device = openni2.Device.open_any()

        # open depth stream
        self._depth_stream = self._device.create_depth_stream()
        self._depth_stream.configure_mode(PrimesenseSensor.DEPTH_IM_WIDTH,
                                          PrimesenseSensor.DEPTH_IM_HEIGHT,
                                          PrimesenseSensor.FPS,
                                          openni2.PIXEL_FORMAT_DEPTH_1_MM) 
        self._depth_stream.start()

        # open color stream
        self._color_stream = self._device.create_color_stream()
        self._color_stream.configure_mode(PrimesenseSensor.COLOR_IM_WIDTH,
                                          PrimesenseSensor.COLOR_IM_HEIGHT,
                                          PrimesenseSensor.FPS,
                                          openni2.PIXEL_FORMAT_RGB888) 
        self._color_stream.camera.set_auto_white_balance(self._auto_white_balance)
        self._color_stream.camera.set_auto_exposure(self._auto_exposure)
        self._color_stream.start()

        # configure device
        if self._registration_mode == PrimesenseRegistrationMode.DEPTH_TO_COLOR:
            self._device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        else:
            self._device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)

        self._device.set_depth_color_sync_enabled(self._enable_depth_color_sync)

        self._running = True

    def stop(self):
        """ Stop the sensor """
        # check that everything is running
        if not self._running or self._device is None:
            logging.warning('Primesense not running. Aborting stop')
            return False

        # stop streams
        if self._depth_stream:
            self._depth_stream.stop()
        if self._color_stream:
            self._color_stream.stop()
        self._running = False
        
        # Unload openni2
        openni2.unload()
        return True

    def _read_depth_image(self):
        """ Reads a depth image from the device """
        # read raw uint16 buffer
        im_arr = self._depth_stream.read_frame()
        raw_buf = im_arr.get_buffer_as_uint16()
        buf_array = np.array([raw_buf[i] for i in range(PrimesenseSensor.DEPTH_IM_WIDTH * PrimesenseSensor.DEPTH_IM_HEIGHT)])

        # convert to image in meters
        depth_image = buf_array.reshape(PrimesenseSensor.DEPTH_IM_HEIGHT,
                                        PrimesenseSensor.DEPTH_IM_WIDTH)
        depth_image = depth_image * MM_TO_METERS # convert to meters
        if self._flip_images:
            depth_image = np.flipud(depth_image)
        else:
            depth_image = np.fliplr(depth_image)
        return DepthImage(depth_image, frame=self._frame)

    def _read_color_image(self):
        """ Reads a color image from the device """
        # read raw buffer
        im_arr = self._color_stream.read_frame()
        raw_buf = im_arr.get_buffer_as_triplet()
        r_array = np.array([raw_buf[i][0] for i in range(PrimesenseSensor.COLOR_IM_WIDTH * PrimesenseSensor.COLOR_IM_HEIGHT)])        
        g_array = np.array([raw_buf[i][1] for i in range(PrimesenseSensor.COLOR_IM_WIDTH * PrimesenseSensor.COLOR_IM_HEIGHT)])        
        b_array = np.array([raw_buf[i][2] for i in range(PrimesenseSensor.COLOR_IM_WIDTH * PrimesenseSensor.COLOR_IM_HEIGHT)])        

        # convert to uint8 image
        color_image = np.zeros([PrimesenseSensor.COLOR_IM_HEIGHT, PrimesenseSensor.COLOR_IM_WIDTH, 3])
        color_image[:,:,0] = r_array.reshape(PrimesenseSensor.COLOR_IM_HEIGHT,
                                             PrimesenseSensor.COLOR_IM_WIDTH)
        color_image[:,:,1] = g_array.reshape(PrimesenseSensor.COLOR_IM_HEIGHT,
                                             PrimesenseSensor.COLOR_IM_WIDTH)
        color_image[:,:,2] = b_array.reshape(PrimesenseSensor.COLOR_IM_HEIGHT,
                                             PrimesenseSensor.COLOR_IM_WIDTH)
        if self._flip_images:
            color_image = np.flipud(color_image.astype(np.uint8))
        else:
            color_image = np.fliplr(color_image.astype(np.uint8))
        return ColorImage(color_image, frame=self._frame)

    def frames(self):
        """Retrieve a new frame from the Kinect and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        color_im = self._read_color_image()
        depth_im = self._read_depth_image()
        return color_im, depth_im, None

    def median_depth_img(self, num_img=1, fill_depth=0.0):
        """Collect a series of depth images and return the median of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The median DepthImage collected from the frames.
        """
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        median_depth = Image.median_images(depths)
        median_depth.data[median_depth.data == 0.0] = fill_depth
        return median_depth

    def min_depth_img(self, num_img=1):
        """Collect a series of depth images and return the min of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The min DepthImage collected from the frames.
        """
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        return Image.min_images(depths)

class VirtualPrimesenseSensor(CameraSensor):
    SUPPORTED_FILE_EXTS = ['.png', '.npy']

    """ Class for a virtual Primesense sensor that uses pre-captured images
    stored to disk instead of actually connecting to a sensor.
    For debugging purposes.
    """
    def __init__(self, path_to_images, frame=None):
        """Create a new virtualized Primesense sensor.

        This requires a directory containing a specific set of files.

        First, the directory must contain a set of images, where each
        image has three files:
        - color_{#}.png
        - depth_{#}.npy
        - ir_{#}.npy
        In these, the {#} is replaced with the integer index for the
        image. These indices should start at zero and increase
        consecutively.

        Second, the directory must contain CameraIntrisnics files
        for the color and ir cameras:
        - {frame}_color.intr
        - {frame}_ir.intr
        In these, the {frame} is replaced with the reference frame
        name that is passed as a parameter to this function.

        Parameters
        ----------
        path_to_images : :obj:`str`
            The path to a directory containing images that the virtualized
            sensor will return.

        frame : :obj:`str`
            The name of the frame of reference in which the sensor resides.
            If None, this will be discovered from the files in the directory.
        """
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
        self._color_ext = '.png'
        for filename in filenames:
            file_root, file_ext = os.path.splitext(filename)
            color_ind = file_root.rfind('color')

            if file_ext in VirtualPrimesenseSensor.SUPPORTED_FILE_EXTS \
               and color_ind != -1:
                self._color_ext = file_ext

            if self._frame is None and file_ext == INTR_EXTENSION and color_ind != -1:
                self._frame = file_root[:color_ind-1]
                self._color_frame = file_root
                self._ir_frame = file_root

        # load color intrinsics
        color_intr_filename = os.path.join(self._path_to_images, '%s_color.intr' %(self._frame))
        self._color_intr = CameraIntrinsics.load(color_intr_filename)
        ir_intr_filename = os.path.join(self._path_to_images, '%s_ir.intr' %(self._frame))
        self._ir_intr = CameraIntrinsics.load(ir_intr_filename)

    @property
    def path_to_images(self):
        """:obj:`str` : The path to a directory containing images that the virtualized
        sensor will return.
        """
        return self._path_to_images

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise.
        """
        return self._running

    @property
    def frame(self):
        """:obj:`str` : The reference frame of the sensor.
        """
        return self._frame

    @property
    def color_frame(self):
        """:obj:`str` : The reference frame of the color sensor.
        """
        return self._color_frame

    @property
    def ir_frame(self):
        """:obj:`str` : The reference frame of the IR sensor.
        """
        return self._ir_frame

    @property
    def color_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Kinect's color camera.
        """
        return self._color_intr

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Kinect's IR camera.
        """
        return self._ir_intr


    def start(self):
        """Starts the Kinect v2 sensor stream.

        In this virtualized sensor, this simply resets the image index to zero.
        Everytime start is called, we start the stream again at the first image.
        """
        self._im_index = 0
        self._running = True

    def stop(self):
        """Stops the Primesense sensor stream.

        Returns
        -------
        bool
            True if the stream was stopped, False if the device was already
            stopped or was not otherwise available.
        """
        if not self._running:
            return False
        self._running = False
        return True

    def frames(self):
        """Retrieve the next frame from the image directory and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Parameters
        ----------
        skip_registration : bool
            If True, the registration step is skipped.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running or if all images in the
            directory have been used.
        """
        if not self._running:
            raise RuntimeError('Primesense device pointing to %s not runnning. Cannot read frames' %(self._path_to_images))

        if self._im_index > self._num_images:
            raise RuntimeError('Primesense device is out of images')

        # read images
        color_filename = os.path.join(self._path_to_images, 'color_%d%s' %(self._im_index, self._color_ext))
        color_im = ColorImage.open(color_filename, frame=self._frame)
        depth_filename = os.path.join(self._path_to_images, 'depth_%d.npy' %(self._im_index))
        depth_im = DepthImage.open(depth_filename, frame=self._frame)
        self._im_index += 1
        return color_im, depth_im, None

    def median_depth_img(self, num_img=1):
        """Collect a series of depth images and return the median of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The median DepthImage collected from the frames.
        """
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        return Image.median_images(depths)

class PrimesenseSensor_ROS(PrimesenseSensor):
    """ ROS-based version of Primesense RGBD sensor interface
    
    Requires starting the openni2 ROS driver and the two stream_image_buffer (image_buffer.py)
    ros services for depth and color images. By default, the class will look for the depth_image buffer
    and color_image buffers under "{frame}/depth/stream_image_buffer" and "{frame}/rgb/stream_image_buffer"
    respectively (within the current ROS namespace).
    
    This can be changed by passing in depth_image_buffer, color_image_buffer (which change where the program
    looks for the buffer services) and depth_absolute, color_absolute (which changes whether the program prepends
    the current ROS namespace).
    """
    def __init__(self, depth_image_buffer= None, depth_absolute=False, color_image_buffer=None, color_absolute=False,
                 flip_images=True, frame=None, staleness_limit=10., timeout=10):  
        self._flip_images = flip_images
        self._frame = frame
        
        self.staleness_limit = staleness_limit
        self.timeout = timeout

        if self._frame is None:
            self._frame = 'primesense'
        self._color_frame = '%s_color' %(self._frame)
        self._ir_frame = self._frame # same as color since we normally use this one
        
        # Set image buffer locations
        self._depth_image_buffer = ('{0}/depth/stream_image_buffer'.format(frame)
                                    if depth_image_buffer == None else depth_image_buffer)
        self._color_image_buffer = ('{0}/rgb/stream_image_buffer'.format(frame)
                                    if color_image_buffer == None else color_image_buffer)
        if not depth_absolute:
            self._depth_image_buffer = rospy.get_namespace() + self._depth_image_buffer
        if not color_absolute:
            self._color_image_buffer = rospy.get_namespace() + self._color_image_buffer
        
    def start(self):
        """For PrimesenseSensor, start/stop by launching/stopping
        the associated ROS services"""
        pass
    def stop(self):
        """For PrimesenseSensor, start/stop by launching/stopping
        the associated ROS services"""
        pass
    
    def _ros_read_images(self, stream_buffer, number, staleness_limit = 10.):
        """ Reads images from a stream buffer
        
        Parameters
        ----------
        stream_buffer : string
            absolute path to the image buffer service
        number : int
            The number of frames to get. Must be less than the image buffer service's
            current buffer size
        staleness_limit : float, optional
            Max value of how many seconds old the oldest image is. If the oldest image
            grabbed is older than this value, a RuntimeError is thrown.
            
            If None, staleness is ignored.
        Returns
        -------
        List of nump.ndarray objects, each one an image
        Images are in reverse chronological order (newest first)
        """
        
        rospy.wait_for_service(stream_buffer, timeout = self.timeout)
        ros_image_buffer = rospy.ServiceProxy(stream_buffer, ImageBuffer)
        ret = ros_image_buffer(number, 1)
        if not staleness_limit == None:
            if ret.timestamps[-1] > staleness_limit:
                raise RuntimeError("Got data {0} seconds old, more than allowed {1} seconds"
                                   .format(ret.timestamps[-1], staleness_limit))
            
        data = ret.data.reshape(ret.data_dim1, ret.data_dim2, ret.data_dim3).astype(ret.dtype)
        
        # Special handling for 1 element, since dstack's behavior is different
        if number == 1:
            return [data]
        return np.dsplit(data, number)

    @property
    def is_running(self):
        """bool : True if the image buffers are running, or false otherwise.
        
        Does this by grabbing one frame with staleness checking
        """
        try:
            self.frames()
        except:
            return False
        return True
    
    def _read_depth_images(self, num_images):
        """ Reads depth images from the device """
        depth_images = self._ros_read_images(self._depth_image_buffer, num_images, self.staleness_limit)
        for i in range(0, num_images):
            depth_images[i] = depth_images[i] * MM_TO_METERS # convert to meters
            if self._flip_images:
                depth_images[i] = np.flipud(depth_images[i])
                depth_images[i] = np.fliplr(depth_images[i])
            depth_images[i] = DepthImage(depth_images[i], frame=self._frame) 
        return depth_images
    def _read_color_images(self, num_images):
        """ Reads color images from the device """
        color_images = self._ros_read_images(self._color_image_buffer, num_images, self.staleness_limit)
        for i in range(0, num_images):
            if self._flip_images:
                color_images[i] = np.flipud(color_images[i].astype(np.uint8))
                color_images[i] = np.fliplr(color_images[i].astype(np.uint8))
            color_images[i] = ColorImage(color_images[i], frame=self._frame) 
        return color_images
    
    def _read_depth_image(self):
        """ Wrapper to maintain compatibility """
        return self._read_depth_images(1)[0]
    def _read_color_image(self):
        """ Wrapper to maintain compatibility """
        return self._read_color_images(1)[0]
        
    def median_depth_img(self, num_img=1, fill_depth=0.0):
        """Collect a series of depth images and return the median of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The median DepthImage collected from the frames.
        """
        depths = self._read_depth_images(num_img)

        median_depth = Image.median_images(depths)
        median_depth.data[median_depth.data == 0.0] = fill_depth
        return median_depth

    def min_depth_img(self, num_img=1):
        """Collect a series of depth images and return the min of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The min DepthImage collected from the frames.
        """
        depths = self._read_depth_images(num_img)

        return Image.min_images(depths)
        
