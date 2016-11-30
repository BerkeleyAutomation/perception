"""
Script to register sensors to a chessboard for the YuMi setup
Authors: Jeff Mahler and Brenton Chu
"""
import IPython
import logging
import numpy as np
import os
import time

from mpl_toolkits.mplot3d import Axes3D

from core import RigidTransform, YamlConfig
from yumipy import YuMiRobot
from perception import CameraChessboardRegister

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    config_filename = 'cfg/tools/register_camera.yaml'
    config = YamlConfig(config_filename)

    # transform into the world frame
    T_cb_world = RigidTransform.load(config['chessboard_tf'])

    # get camera sensor object
    for sensor_frame, sensor_config in config['sensors'].iteritems():
        T_camera_cb, cb_points_cam, points_3d_plane = CameraChessboardRegister.get_camera_chessboard(sensor_frame, sensor_config['type'],
                                                                sensor_config['device_num'], config['chessboard_registration'])
        T_camera_world = T_cb_world * T_camera_cb

        logging.info('Final Result for sensor %s' %(sensor_frame))
        logging.info('Rotation: ')
        logging.info(T_camera_world.rotation)
        logging.info('Translation: ')
        logging.info(T_camera_world.translation)

        # save tranformation arrays based on setup
        output_dir = os.path.join(config['calib_dir'], sensor_frame)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pose_filename = os.path.join(output_dir, '%s_registration.tf' %(sensor_frame))
        T_camera_world.save(pose_filename)
        f = open(os.path.join(output_dir, 'corners_cb_%s.npy' %(sensor_frame)), 'w')
        np.save(f, points_3d_plane.data)

        # move the robot to the chessboard center for verification
        if config['use_robot']:

            # find the rightmost and further cb point in world frame
            cb_points_world = T_camera_world * cb_points_cam
            cb_point_data_world = cb_points_world.data
            dir_world = np.array([1.0, -1.0, 0])
            dir_world = dir_world / np.linalg.norm(dir_world)
            ip = dir_world.dot(cb_point_data_world)
            target_ind = np.where(ip == np.max(ip))[0]
            target_pt_world = cb_points_world[target_ind[0]]

            # create robot pose relative to target point
            R_gripper_world = np.array([[1.0, 0, 0],
                                     [0, -1.0, 0],
                                     [0, 0, -1.0]])
            t_gripper_world = np.array([target_pt_world.x + config['gripper_offset_x'],
                                        target_pt_world.y + config['gripper_offset_y'],
                                        target_pt_world.z + config['gripper_offset_z']])
            T_gripper_world = RigidTransform(rotation=R_gripper_world,
                                             translation=t_gripper_world,
                                             from_frame='gripper',
                                             to_frame='cb')
            logging.info('Moving robot to point x=%f, y=%f, z=%f' %(t_gripper_world[0], t_gripper_world[1], t_gripper_world[2]))

            # move robot to pose
            y = YuMiRobot()
            y.reset_home()
            y.right.close_gripper()
            time.sleep(1)

            T_lift = RigidTransform(translation=(0,0,0.1), from_frame='cb', to_frame='cb')
            T_gripper_world_lift = T_lift * T_gripper_world
            y.right.goto_pose(T_gripper_world_lift)
            y.right.goto_pose(T_gripper_world)

            # wait for human measurement
            yesno = raw_input('Take measurement. Hit [ENTER] when done')
            y.right.goto_pose(T_gripper_world_lift)
            y.reset_home()
            y.open_grippers()
            y.stop()
