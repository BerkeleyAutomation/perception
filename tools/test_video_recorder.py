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

from perception import VideoRecorder

if __name__ == '__main__':
    recorder = VideoRecorder(0)
    recorder.start()
    recorder.start_recording('test.avi')
    time.sleep(10)
    recorder.stop_recording()
    recorder.stop()
