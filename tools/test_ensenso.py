"""
Tests the ensenso.
Author: Jeff Mahler
"""
import logging
import rospy
import sys
import time

from perception import EnsensoSensor, PhoXiSensor
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

def main(args):
    # set logging
    logging.getLogger().setLevel(logging.INFO)
    rospy.init_node('ensenso_reader', anonymous=True)

    num_frames = 10
    #sensor = PhoXiSensor(frame='phoxi',
    #                     size='small')
    sensor = EnsensoSensor(frame='ensenso')
    sensor.start()

    total_time = 0
    for i in range(num_frames):        
        if i > 0:
            start_time = time.time()

        _, depth_im, _ = sensor.frames()

        if i > 0:
            total_time += time.time() - start_time
            print('Frame %d' %(i))
            print('Avg FPS: %.5f' %(float(i) / total_time))
        
    depth_im = sensor.median_depth_img(num_img=5)
    point_cloud = sensor.ir_intrinsics.deproject(depth_im) 
    point_cloud.remove_zero_points()

    sensor.stop()

    vis2d.figure()
    vis2d.imshow(depth_im)
    vis2d.title('PhoXi - Raw')
    vis2d.show()

    vis3d.figure()
    vis3d.points(point_cloud, random=True, subsample=10, scale=0.0025)
    vis3d.show()
    
if __name__ == '__main__':
    main(sys.argv)
