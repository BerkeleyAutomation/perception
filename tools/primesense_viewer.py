"""
Simple tool to view output and fps of a primesense RGBD sensor
Author: Jeff Mahler
"""
import IPython
import logging
import numpy as np
import time

from perception import PrimesenseSensor

from visualization import Visualizer2D as vis
from visualization import Visualizer3D as vis3d

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    sensor = PrimesenseSensor()

    logging.info('Starting sensor')
    sensor.start()
    camera_intr = sensor.ir_intrinsics

    n = 15
    frame_rates = []
    for i in range(n):
        logging.info('Reading frame %d of %d' %(i+1, n))
        read_start = time.time()
        color_im, depth_im, _ = sensor.frames()
        read_stop = time.time()
        frame_rates.append(1.0/(read_stop-read_start))

    logging.info('Avg fps: %.3f' %(np.mean(frame_rates)))

    color_im = color_im.inpaint(rescale_factor=0.5)
    depth_im = depth_im.inpaint(rescale_factor=0.5)
    point_cloud = camera_intr.deproject(depth_im)

    vis3d.figure()
    vis3d.points(point_cloud, subsample=15)
    vis3d.show()

    vis.figure()
    vis.subplot(1,2,1)
    vis.imshow(color_im)
    vis.subplot(1,2,2)
    vis.imshow(depth_im)
    vis.show()

    sensor.stop()

