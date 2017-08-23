"""
Quick tool to test video recording.
Author: Jeff Mahler
"""
import cv2
import IPython
import numpy as np
import os
import sys
import time

from perception import OpenCVCameraSensor, VideoRecorder
from visualization import Visualizer2D as vis

if __name__ == '__main__':
    sensor = OpenCVCameraSensor(0)
    sensor.start()
    im = sensor.frames()
    sensor.stop()
    
    #vis.figure()
    #vis.imshow(im)
    #vis.show()
    IPython.embed()

    recorder = VideoRecorder(0, fps=10, res=(640,480))
    recorder.start()
    recorder.start_recording('test.avi')
    time.sleep(10)
    recorder.stop_recording()
    recorder.stop()
