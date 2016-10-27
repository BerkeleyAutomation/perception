'''
Class to record videos from webcams using opencv
Author: Jacky
'''
from multiprocessing import Process, Queue
import cv2

class _Camera(Process):

    def __init__(self, camera, cmd_q, res, codec, fps):
        Process.__init__(self)
        
        self.res = res
        self.fps = fps
        self.codec = codec

        self.camera = camera
        self.fourcc = cv2.cv.CV_FOURCC(*self.codec)
        
        self.cmd_q = cmd_q
        self.recording = False
        self.out = None

    def run(self):
        while True:
            if not self.cmd_q.empty():
                cmd = self.cmd_q.get()
                if cmd[0] == 'stop':
                    self.out.release()
                    self.recording = False
                elif cmd[0] == 'start':
                    filename = cmd[1]
                    self.out = cv2.VideoWriter(filename, self.fourcc, self.fps, self.res)                    
                    if not self.out.isOpened():
                        raise Exception("Unable to open video writer for file {0}, codec {1}, at fps {2}, res {3}".format(
                                                                            filename, self.codec, self.fps, self.res))
                    self.recording = True
                    
            if self.recording:
                ret_val, frame = self.camera.read()
                if ret_val:
                    self.out.write(frame)

class VideoRecorder:

    def __init__(self, device_id, res=(640, 480), codec='XVID', fps=30):
        '''
        Create video recorder object.
        
        Args:
            device_id: index of device
            res: resolution of recording and saving. defaults to (640, 480)
            codec: codec used for encoding video. default to XVID. 
            fps: fps of vide captures. defaults to 30
        '''
        self._res = res
        self._codec = codec
        self._fps = fps
        
        self._cmd_q = Queue()
        
        self._actual_camera = None
        self._actual_camera = cv2.VideoCapture(device_id)

        if not self._actual_camera.isOpened():
            raise Exception("Unable to open video device for video{0}".format(device_id))
        
        self._recording = False
        self._started = False
        
    def start(self):
        self._started = True
        self._camera = _Camera(self._actual_camera, self._cmd_q, self._res, self._codec, self._fps)
        self._camera.start()

    def start_recording(self, output_file):
        if not self._started:
            raise Exception("Must start the video recorder first by calling .start()!")
        if self._recording:
            raise Exception("Cannot record a video while one is already recording!")
        self._recording = True
        self._cmd_q.put(('start', output_file))
        
    def stop_recording(self):
        if not self._recording:
            raise Exception("Cannot stop a video recording when it's not recording!")
        self._cmd_q.put(('stop',))
        self._recording = False

    def stop(self):
        if not self._started:
            raise Exception("Cannot stop a video recorder before starting it!")
        self._started = False
        self._actual_camera.release()
        self._camera.terminate()