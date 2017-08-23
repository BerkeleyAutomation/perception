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

from perception import OpenCVCameraSensor, PrimesenseSensor_ROS, VideoRecorder
from visualization import Visualizer2D as vis

if __name__ == '__main__':
    """
    num_frames = 30
    data = None
    writer = si.FFmpegWriter('test.mp4')

    sensor = OpenCVCameraSensor(0)
    sensor.start()

    for i in range(num_frames):
        print 'Frame', i
        im = sensor.frames()
        if data is None:
            data = np.zeros([num_frames, im.height,
                             im.width, im.channels]).astype(np.uint8)
        data[i,...] = im.raw_data
        writer.writeFrame(data[i:i+1,...])
    sensor.stop()
    writer.close()
    exit(0)
    """

    #vis.figure()
    #vis.imshow(im)
    #vis.show()
    #si.vwrite('test.mp4', data)

    sensor = PrimesenseSensor_ROS(frame='primesense_overhead')
    sensor.start()

    recorder = VideoRecorder(sensor)
    print 'recording'
    recorder.start()
    recorder.start_recording('test.mp4')
    time.sleep(10)
    recorder.stop_recording()
    recorder.stop()

    sensor.stop()
