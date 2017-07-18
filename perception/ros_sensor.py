"""
Generic ros-based sensor class
"""
import logging
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import multiprocessing, Queue
import os, sys
from perception import CameraSensor, ColorImage, DepthImage, IrImage

# TODO:
# Giving a warning if stale data is being returned/delete stale data
# Maybe add sleep between main loop runs (if it isn't hogging cpu cycles then eh)
class _ImageBuffer(multiprocessing.Process):
    def __init__(self, instream, encoding="passthrough", absolute=False, bufsize=100):
        '''Initializes an image buffer process.
        This uses a subprocess used to buffer a ROS image stream.

        Parameters
        ----------
            instream : string
                    ROS image stream to buffer
            absolute : bool, optional
                    if True, current frame is not prepended to instream (default False)
            bufsize : int, optional
                    Maximum size of image buffer (number of images stored, default 100)
            encoding : String, optional
                    Encoding to output in
        '''
        multiprocessing.Process.__init__(self)
        
        # Initialize input and output queues
        self._req_q = multiprocessing.Queue()
        self._res_q = multiprocessing.Queue()
        
        self.bufsize = bufsize
        self.encoding=encoding
        if absolute:
            self.stream_to_buffer = instream
        else:
            self.stream_to_buffer = rospy.get_namespace() + instream
            

    def run(self):
        # Initialize the node. Anonymous to allow dupes.
        rospy.init_node('stream_image_buffer', anonymous = True)
        
        # Initialize the CvBridge and image buffer list, as well as misc counting things
        bridge = CvBridge()
        buffer_list = []
        # Create callback function for subscribing
        def callback(data):
            """Callback function for subscribing to an Image topic and creating a buffer
            """ 
            # Get cv image (which is a numpy array) from data
            cv_image = bridge.imgmsg_to_cv2(data, desired_encoding=self.encoding)
            # Insert and roll buffer
            buffer_list.insert(0, (cv_image, rospy.get_time()))
            if(len(buffer_list) > self.bufsize):
                buffer_list.pop()
        # Initialize subscriber with our callback
        rospy.Subscriber(self.stream_to_buffer, Image, callback)
        
        # Main loop
        while True:
            try:
                try: 
                    req = self._req_q.get(block = False)
                    if req == "TERM":
                        return
                    # If this works we put a return with status 0
                    self._res_q.put((0, self._handle_req(buffer_list, *req)))
                except Queue.Empty:
                    pass
                # On RuntimeError, we pass it back to parent process to throw
                except RuntimeError as e:
                    self._res_q.put((1, e))
                # Parent pid gets set to 1 when parent dies, so we kill if that's the case
                if os.getppid() == 1:
                    sys.exit(0)
            except KeyboardInterrupt:
                continue
            
    def _handle_req(self, buffer_list, num_requested, timing_mode):
        """Handles a request for images. Private method, used in child buffer process
        
        Parameters
        ----------
            buffer_list : list
                list to pull buffered data from
            num_requested : int
                Number of image-timestamp pairs to return
            timing_mode : string
                One of {"absolute, "relative"}. absolute returns UNIX timestamps,
                relative returns age at request received.
                Everything is floating point, in seconds
        Returns
        -------
        out :
            List of image-timestamp pairs. Images are np arrays
        """
        # Register time of request
        req_time = rospy.get_time()
        
        # Check if request fits in buffer
        if num_requested > len(buffer_list):
            raise RuntimeError("Number of images requested exceeds current buffer size")
        
        # Cut out the images and timestamps we're returning, save image shape
        ret_images, ret_times = zip(*buffer_list[:num_requested])
        
        # Get timestamps in desired mode
        if timing_mode == "absolute":
            ret_times = np.asarray(ret_times)
        elif timing_mode == "relative":
            ret_times = np.asarray([req_time - time for time in ret_times])
        else:
            raise RuntimeError("{0} is not a value for timing_mode".format(timing_mode))
        # Re-zip into image-time pairs and return
        return [pair for pair in zip(ret_images, ret_times)]
    
    def request_images(self, num_requested, timing_mode="absolute"):
        """Sends a request for images. Used in parent process
        
        Parameters
        ----------
            num_requested : int
                Number of images to pull from buffer. Most recent images are returned
            timing_mode : string
                One of {"absolute, "relative"}. absolute returns UNIX timestamps,
                relative returns age at request received.
                Everything is floating point, in seconds
        Returns
        -------
        out :
            List of image-timestamp pairs. Images are np arrays
        """
        self._req_q.put((num_requested, timing_mode))
        try:
            result = self._res_q.get(timeout=10)
        except Queue.Empty:
            raise RuntimeError("Request has timed out")
        if result[0] == 0:
            return result[1]
        else:
            raise result[1]
    
    def terminate(self):
        """Kill the process. Since it ignores sigterm we do it this way.
        """
        self._req_q.put("TERM")
        
class _DummyImageBuffer(object):
    """Initializes a dummy image buffer that returns None
    """
    def __init__(self):
        pass
    def request_images(self, num_requested, timing_mode="absolute"):
        return None
    def start(self):
        pass
    def terminate(self):
        pass

# TODO
# Camera intrinsics
class RosSensor(CameraSensor):
    """ Class for a general ROS-based camera 
    """
    def __init__(self, frame, rgb_stream, depth_stream, ir_stream, absolute=False):
        """Instantiate a ROS stream-based sensor
        """
        self._frame = frame
        
        self.rgb_stream   = _DummyImageBuffer()
        self.ir_stream    = _DummyImageBuffer()
        self.depth_stream = _DummyImageBuffer()
        
        if rgb_stream is not None:
            self.rgb_stream = _ImageBuffer(rgb_stream, absolute=absolute, encoding="rgb8")
        if ir_stream is not None:
            self.ir_stream = _ImageBuffer(ir_stream, absolute=absolute)
        if depth_stream is not None:
            self.depth_stream = _ImageBuffer(depth_stream, absolute=absolute)
            
    def start(self):
        """Starts the subscriber processes
        """
        self.rgb_stream.start()
        self.ir_stream.start()
        self.depth_stream.start()

    def stop(self):
        """Stops the sensor stream.
        """
        self.rgb_stream.terminate()
        self.ir_stream.terminate()
        self.depth_stream.terminate()

    def frames(self):
        """Returns the latest set of frames.
        """
        color_image = self.rgb_stream.request_images(1)
        color_image = None if color_image is None else ColorImage(color_image[0][0], frame=self._frame)
        
        ir_image = self.ir_stream.request_images(1)
        ir_image = (None if ir_image is None else
                    IrImage(np.array(ir_image[0][0], dtype=np.uint8), frame=self._frame))
        
        depth_image = self.depth_stream.request_images(1)
        depth_image = (None if depth_image is None else
                       DepthImage(np.array(depth_image[0][0], dtype=np.float32), frame=self._frame))
        return color_image, depth_image, ir_image
        