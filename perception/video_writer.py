'''
Simple tool to write a series of images into a video.
'''

import cv2
from image import Image
from multiprocessing import Process

def write_video(frames, output_file, codec='XVID', fps=30, blocking=True):

    def write():
        res = frames[0].shape[:2][::-1]
        fourcc = cv2.cv.CV_FOURCC(*codec)
        out = cv2.VideoWriter(output_file, fourcc, fps, res)

        if not out.isOpened():
            raise Exception("Could not open video writer for file {0}".format(output_file))

        for frame in frames:
            if isinstance(frame, Image):
                out.write(frame.raw_data)
            else:
                out.write(frame)

        out.release()

    if blocking:
        write()
    else:
        p = Process(target=write)
        p.start()
