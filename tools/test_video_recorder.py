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

import skvideo.io as si

from perception import WebcamSensor, PrimesenseSensor_ROS, VideoRecorder
from visualization import Visualizer2D as vis

if __name__ == '__main__':
    sensor = WebcamSensor(device_id=0)

    recorder = VideoRecorder(sensor)
    print 'recording'
    recorder.start()
    recorder.start_recording('test.mp4')
    time.sleep(10)
    recorder.stop_recording()
    recorder.stop()

    sensor.stop()
