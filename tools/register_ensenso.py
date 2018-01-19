"""
Register Ensenso to the robot
Author: Jeff Mahler
"""
import argparse
import IPython
import logging
import numpy as np
import os
import struct
import sys
import time

from cv_bridge import CvBridge, CvBridgeError
from ensenso.srv import CalibrateHandEye, CollectPattern, EstimatePatternPose
from geometry_msgs.msg import Pose, PoseArray
import rospy

from autolab_core import Point, RigidTransform, YamlConfig
from perception import DepthImage, EnsensoSensor
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d
from yumipy import YuMiRobot
from yumipy import YuMiConstants as YMC

def register_ensenso(config):
    # set parameters
    average = True
    add_to_buffer = True
    clear_buffer = False
    decode = True
    grid_spacing = -1

    # read parameters
    num_patterns = config['num_patterns']
    max_tries = config['max_tries']
    
    # load tf pattern to world
    T_pattern_world = RigidTransform.load(config['pattern_tf'])
    
    # initialize sensor
    sensor_frame = config['sensor_frame']
    sensor = EnsensoSensor(sensor_frame)

    # initialize node
    rospy.init_node('register_ensenso', anonymous=True)
    rospy.wait_for_service('/%s/collect_pattern' %(sensor_frame))
    rospy.wait_for_service('/%s/estimate_pattern_pose' %(sensor_frame))
    
    # start sensor
    print('Starting sensor')
    sensor.start()
    time.sleep(1)
    print('Sensor initialized')
    
    # perform registration
    try:
        print('Collecting patterns')
        num_detected = 0
        i = 0
        while num_detected < num_patterns and i < max_tries:
            collect_pattern = rospy.ServiceProxy('/%s/collect_pattern' %(sensor_frame), CollectPattern)
            resp = collect_pattern(add_to_buffer,
                                   clear_buffer,
                                   decode,
                                   grid_spacing)
            if resp.success:
                print('Detected pattern %d' %(num_detected))
                num_detected += 1
            i += 1

        if i == max_tries:
            raise ValueError('Failed to detect calibration pattern!')
            
        print('Estimating pattern pose')
        estimate_pattern_pose = rospy.ServiceProxy('/%s/estimate_pattern_pose' %(sensor_frame), EstimatePatternPose)
        resp = estimate_pattern_pose(average)

        q_pattern_camera = np.array([resp.pose.orientation.w,
                                     resp.pose.orientation.x,
                                     resp.pose.orientation.y,
                                     resp.pose.orientation.z])
        t_pattern_camera = np.array([resp.pose.position.x,
                                     resp.pose.position.y,
                                     resp.pose.position.z])
        T_pattern_camera = RigidTransform(rotation=q_pattern_camera,
                                          translation=t_pattern_camera,
                                          from_frame='pattern',
                                          to_frame=sensor_frame)
    except rospy.ServiceException as e:
        print('Service call failed: %s' %(str(e)))

    # compute final transformation
    T_camera_world = T_pattern_world * T_pattern_camera.inverse()
    
    # save final transformation
    output_dir = os.path.join(config['calib_dir'], sensor_frame)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pose_filename = os.path.join(output_dir, '%s_to_world.tf' %(sensor_frame))
    T_camera_world.save(pose_filename)
    intr_filename = os.path.join(output_dir, '%s.intr' %(sensor_frame))
    sensor.ir_intrinsics.save(intr_filename)

    # log final transformation
    print('Final Result for sensor %s' %(sensor_frame))
    print('Rotation: ')
    print(T_camera_world.rotation)
    print('Quaternion: ')
    print(T_camera_world.quaternion)
    print('Translation: ')
    print(T_camera_world.translation)

    # stop sensor
    sensor.stop()

    # move robot to calib pattern center
    if config['use_robot']:  
        # create robot pose relative to target point
        target_pt_camera = Point(T_pattern_camera.translation, sensor_frame)
        target_pt_world = T_camera_world * target_pt_camera
        R_gripper_world = np.array([[1.0, 0, 0],
                                    [0, -1.0, 0],
                                    [0, 0, -1.0]])
        t_gripper_world = np.array([target_pt_world.x + config['gripper_offset_x'],
                                    target_pt_world.y + config['gripper_offset_y'],
                                    target_pt_world.z + config['gripper_offset_z']])
        T_gripper_world = RigidTransform(rotation=R_gripper_world,
                                         translation=t_gripper_world,
                                         from_frame='gripper',
                                         to_frame='world')

        # move robot to pose
        y = YuMiRobot(tcp=YMC.TCP_SUCTION_STIFF)
        y.reset_home()
        time.sleep(1)

        T_lift = RigidTransform(translation=(0,0,0.05), from_frame='world', to_frame='world')
        T_gripper_world_lift = T_lift * T_gripper_world

        y.right.goto_pose(T_gripper_world_lift)
        y.right.goto_pose(T_gripper_world)
        
        # wait for human measurement
        yesno = raw_input('Take measurement. Hit [ENTER] when done')
        y.right.goto_pose(T_gripper_world_lift)
        y.reset_home()
        y.stop()
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Register a camera to a robot')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/register_ensenso.yaml', help='filename of a YAML configuration for registration')
    args = parser.parse_args()
    config_filename = args.config_filename
    config = YamlConfig(config_filename)

    # perform registration
    register_ensenso(config)
