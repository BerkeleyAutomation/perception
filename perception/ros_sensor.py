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

# TODO:
# Giving a warning if stale data is being returned/delete stale data
# Maybe add sleep between main loop runs (if it isn't hogging cpu cycles then eh)
# Write a CameraSensor class based on this (this is the big one)
class _ImageBuffer(multiprocessing.Process):
    def __init__(self, instream, absolute=False, bufsize=100):
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
        '''
        multiprocessing.Process.__init__(self)
        
        # Initialize input and output queues
        self._req_q = multiprocessing.Queue()
        self._res_q = multiprocessing.Queue()
        
        self.bufsize = bufsize
        if absolute:
            self.stream_to_buffer = instream
        else:
            self.stream_to_buffer = rospy.get_namespace() + instream
            

    def run(self):
        # Initialize the node. Anonymous to allow dupes.
        rospy.init_node('stream_image_buffer', anonymous = True)
        
        # Initialize the CvBridge and image buffer list, as well as misc counting things
        bridge = CvBridge()
        buffer = []
        # Create callback function for subscribing
        def callback(data):
            """Callback function for subscribing to an Image topic and creating a buffer
            """ 
            # Get cv image (which is a numpy array) from data
            cv_image = bridge.imgmsg_to_cv2(data)
            # Insert and roll buffer
            buffer.insert(0, (cv_image, rospy.get_time()))
            if(len(buffer) > self.bufsize):
                buffer.pop()
        # Initialize subscriber with our callback
        rospy.Subscriber(self.stream_to_buffer, Image, callback)
        
        # Main loop
        while True:
            try:
                try: 
                    req = self._req_q.get(block = False)
                    # If this works we put a return with status 0
                    self._res_q.put((0, self._handle_req(buffer, *req)))
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
            
    def _handle_req(self, buffer, num_requested, timing_mode):
        """Handles a request for images. Private method, used in child buffer process
        
        Parameters
        ----------
            buffer : list
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
        if num_requested > len(buffer):
            raise RuntimeError("Number of images requested exceeds current buffer size")
        
        # Cut out the images and timestamps we're returning, save image shape
        ret_images, ret_times = zip(*buffer[:num_requested])
        
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
        