"""
Test the VideoRecorder class
Author: Mike Danielczuk
"""
import time

from perception import VideoRecorder

if __name__ == "__main__":
    v = VideoRecorder()
    v.start_recording("test_video.mp4")
    time.sleep(5.0)
    v.stop_recording()
