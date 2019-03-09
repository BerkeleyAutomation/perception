"""
Script to register sensors to a chessboard for the YuMi setup
Authors: Jeff Mahler and Brenton Chu
""" 
import argparse
import cv2
import IPython
import logging
import numpy as np
import os
import sys
import time
import traceback

from mpl_toolkits.mplot3d import Axes3D

import rospy
import rosgraph.roslogging as rl

from autolab_core import Point, PointCloud, RigidTransform, YamlConfig
from perception import CameraChessboardRegistration, RgbdSensorFactory

from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d
from yumipy import YuMiRobot
from yumipy import YuMiConstants as YMC

global clicked_pt
clicked_pt = None
pt_radius = 2
pt_color = (255,0,0)
def click_gripper(event, x, y, flags, param):
    global clicked_pt
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked_pt = np.array([x,y])
        logging.info('Clicked: {}'.format(clicked_pt))
        
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Register a camera to a robot')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/register_camera.yaml', help='filename of a YAML configuration for registration')
    args = parser.parse_args()
    config_filename = args.config_filename
    config = YamlConfig(config_filename)
    
    # get known tf from chessboard to world
    T_cb_world = RigidTransform.load(config['chessboard_tf'])
    
    # initialize node
    rospy.init_node('register_camera', anonymous=True)
    logging.getLogger().addHandler(rl.RosStreamHandler())

    # get camera sensor object
    for sensor_frame, sensor_data in config['sensors'].iteritems():
        logging.info('Registering %s' %(sensor_frame))
        sensor_config = sensor_data['sensor_config']
        registration_config = sensor_data['registration_config'].copy()
        registration_config.update(config['chessboard_registration'])
        
        # open sensor
        try:
            sensor_type = sensor_config['type']
            sensor_config['frame'] = sensor_frame
            logging.info('Creating sensor')
            sensor = RgbdSensorFactory.sensor(sensor_type, sensor_config)
            logging.info('Starting sensor')
            sensor.start()
            ir_intrinsics = sensor.ir_intrinsics
            logging.info('Sensor initialized')

            # register
            reg_result = CameraChessboardRegistration.register(sensor, registration_config)
            T_camera_world = T_cb_world * reg_result.T_camera_cb

            logging.info('Final Result for sensor %s' %(sensor_frame))
            logging.info('Rotation: ')
            logging.info(T_camera_world.rotation)
            logging.info('Quaternion: ')
            logging.info(T_camera_world.quaternion)
            logging.info('Translation: ')
            logging.info(T_camera_world.translation)

        except Exception as e:
            logging.error('Failed to register sensor {}'.format(sensor_frame))
            traceback.print_exc()
            continue

        # fix the chessboard corners
        if config['fix_orientation_cb_to_world']:
            # read params
            num_pts_x = config['grid_x']
            num_pts_y = config['grid_y']
            grid_width = config['grid_width']
            grid_height = config['grid_height']
            gripper_height = config['gripper_height']
            grid_center_x = config['grid_center_x']
            grid_center_y = config['grid_center_y']
            
            # determine robot poses
            robot_poses = []
            for i in range(num_pts_x):
                x = -float(grid_width) / 2 + grid_center_x + float(i * grid_width) / num_pts_x
                for j in range(num_pts_y):
                    y = -float(grid_height) / 2 + grid_center_y + float(j * grid_height) / num_pts_y

                    # form robot pose
                    R_robot_world = np.array([[1, 0, 0],
                                              [0, 0, 1],
                                              [0, -1, 0]])
                    t_robot_world = np.array([x, y, gripper_height])
                    T_robot_world = RigidTransform(rotation=R_robot_world,
                                                   translation=t_robot_world,
                                                   from_frame='gripper',
                                                   to_frame='world')
                    robot_poses.append(T_robot_world)

            # start robot
            y = YuMiRobot(tcp=YMC.TCP_SUCTION_STIFF)
            y.set_z('fine')
            y.reset_home()
            global clicked_pt
            
            # iteratively go to poses
            robot_points_camera = []
            for robot_pose in robot_poses:
                # reset clicked pt
                clicked_pt = None

                # move to pose
                y.right.goto_pose(robot_pose, wait_for_res=True)
                    
                # capture image
                color_im, depth_im, _ = sensor.frames()
                depth_im = depth_im.inpaint(0.25)
                cv2.namedWindow('click')
                cv2.setMouseCallback('click', click_gripper)
                while True:
                    if clicked_pt is None:
                        cv2.imshow('click', color_im.data)
                    else:
                        im = color_im.data.copy()
                        cv2.circle(im,
                                   tuple(clicked_pt.tolist()),
                                   pt_radius,
                                   pt_color,
                                   -1)
                        cv2.imshow('click', im)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') and clicked_pt is not None:
                        logging.info('Moving to next pose')
                        break

                # store clicked pt in 3D
                logging.info('Point collection complete')
                depth = depth_im[clicked_pt[1], clicked_pt[0]]
                p = ir_intrinsics.deproject_pixel(depth,
                                                  Point(clicked_pt, frame=ir_intrinsics.frame))

                robot_points_camera.append(p.data)

            # reset
            y.reset_home()
            y.stop()
            
            # collect
            true_robot_points_world = PointCloud(np.array([T.translation for T in robot_poses]).T,
                                                 frame=ir_intrinsics.frame)
            est_robot_points_world = T_camera_world * PointCloud(np.array(robot_points_camera).T,
                                                                 frame=ir_intrinsics.frame)
            mean_true_robot_point = np.mean(true_robot_points_world.data, axis=1).reshape(3,1)
            mean_est_robot_point = np.mean(est_robot_points_world.data, axis=1).reshape(3,1)

            # fit a plane
            best_R_cb_world = None
            best_dist = np.inf
            k = 0
            K = 25
            num_poses = len(robot_poses)
            sample_size = int(num_poses * 0.3)
            min_inliers = int(num_poses * 0.6)
            dist_thresh = 0.0015
            true_robot_points_world._data = true_robot_points_world._data - mean_true_robot_point
            est_robot_points_world._data = est_robot_points_world._data - mean_est_robot_point
            while k < K:
                ind = np.random.choice(num_poses, size=sample_size, replace=False)
                H = est_robot_points_world.data[:,ind].dot(true_robot_points_world.data[:,ind].T)
                U, S, V = np.linalg.svd(H)
                R_cb_world = V.T.dot(U.T)
                
                fixed_robot_points_world = R_cb_world.dot(est_robot_points_world.data)
                diffs = fixed_robot_points_world - true_robot_points_world.data
                dists = np.linalg.norm(diffs, axis=0)
                inliers = np.where(dists < dist_thresh)[0]
                num_inliers = inliers.shape[0]

                print k, num_inliers, np.mean(dists)

                if num_inliers >= min_inliers:
                    ind = inliers
                    H = est_robot_points_world.data[:,ind].dot(true_robot_points_world.data[:,ind].T)
                    U, S, V = np.linalg.svd(H)
                    R_cb_world = V.T.dot(U.T)
                
                    fixed_robot_points_world = R_cb_world.dot(est_robot_points_world.data)
                    diffs = fixed_robot_points_world - true_robot_points_world.data
                    dists = np.linalg.norm(diffs, axis=0)

                    mean_dist = np.mean(dists[ind])
                    if mean_dist < best_dist:
                        best_dist = mean_dist
                        best_R_cb_world = R_cb_world
                k += 1
                        
            R_cb_world = best_R_cb_world
            T_corrected_cb_world = RigidTransform(rotation=R_cb_world,
                                                  from_frame='world',
                                                  to_frame='world')
            R_cb_world = R_cb_world.dot(T_cb_world.rotation)
            T_cb_world = RigidTransform(rotation=R_cb_world,
                                        translation=T_cb_world.translation,
                                        from_frame=T_cb_world.from_frame,
                                        to_frame=T_cb_world.to_frame)
            T_camera_world = T_cb_world * reg_result.T_camera_cb
            T_cb_world.save(config['chessboard_tf'])
            
            # vis
            if config['vis_points']:
                _, depth_im, _ = sensor.frames()
                points_world = T_camera_world * ir_intrinsics.deproject(depth_im)
                true_robot_points_world = PointCloud(np.array([T.translation for T in robot_poses]).T,
                                                     frame=ir_intrinsics.frame)
                est_robot_points_world = T_camera_world * PointCloud(np.array(robot_points_camera).T,
                                                                     frame=ir_intrinsics.frame)
                mean_est_robot_point = np.mean(est_robot_points_world.data, axis=1).reshape(3,1)
                est_robot_points_world._data = est_robot_points_world._data - mean_est_robot_point + mean_true_robot_point
                fixed_robot_points_world = T_corrected_cb_world * est_robot_points_world
                mean_fixed_robot_point = np.mean(fixed_robot_points_world.data, axis=1).reshape(3,1)
                fixed_robot_points_world._data = fixed_robot_points_world._data - mean_fixed_robot_point + mean_true_robot_point
                vis3d.figure()
                vis3d.points(points_world, color=(0,1,0), subsample=10, random=True, scale=0.001)
                vis3d.points(true_robot_points_world, color=(0,0,1), scale=0.001)
                vis3d.points(fixed_robot_points_world, color=(1,1,0), scale=0.001)
                vis3d.points(est_robot_points_world, color=(1,0,0), scale=0.001)
                vis3d.pose(T_camera_world)
                vis3d.show()

        # save tranformation arrays based on setup
        output_dir = os.path.join(config['calib_dir'], sensor_frame)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pose_filename = os.path.join(output_dir, '%s_to_world.tf' %(sensor_frame))
        T_camera_world.save(pose_filename)
        intr_filename = os.path.join(output_dir, '%s.intr' %(sensor_frame))
        ir_intrinsics.save(intr_filename)
        f = open(os.path.join(output_dir, 'corners_cb_%s.npy' %(sensor_frame)), 'w')
        np.save(f, reg_result.cb_points_cam.data)
                
        # move the robot to the chessboard center for verification
        if config['use_robot']:  
            # get robot type
            robot_type = 'yumi'
            if 'robot_type' in config.keys():
                robot_type = config['robot_type']
            
            # find the rightmost and further cb point in world frame
            cb_points_world = T_camera_world * reg_result.cb_points_cam
            cb_point_data_world = cb_points_world.data
            dir_world = np.array([-1.0, 1.0, 0])
            dir_world = dir_world / np.linalg.norm(dir_world)
            ip = dir_world.dot(cb_point_data_world)

            # open interface to robot
            if robot_type == 'ur5':
                from ur_control import UniversalRobot, ToolState, T_KINEMATIC_AVOIDANCE_WORLD, KINEMATIC_AVOIDANCE_JOINTS
                robot = UniversalRobot()
                robot.reset_home()
            else:
                y = YuMiRobot(tcp=YMC.TCP_SUCTION_STIFF)
                y.reset_home()
                robot = y.right
                waypoints = []
            time.sleep(1)
                
            # choose target point #1
            target_ind = np.where(ip == np.max(ip))[0]
            target_pt_world = cb_points_world[target_ind[0]]
                
            # create robot pose relative to target point
            if robot_type == 'ur5':
                R_gripper_world = np.array([[-1.0, 0, 0],
                                            [0, 1.0, 0],
                                            [0, 0, -1.0]])
            else:
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

            T_lift = RigidTransform(translation=(0,0,0.05), from_frame='cb', to_frame='cb')
            T_gripper_world_lift = T_lift * T_gripper_world
            T_orig_gripper_world_lift = T_gripper_world_lift.copy()

            if config['vis_cb_corners']:
                _, depth_im, _ = sensor.frames()
                points_world = T_camera_world * ir_intrinsics.deproject(depth_im)
                vis3d.figure()
                vis3d.points(cb_points_world, color=(0,0,1), scale=0.005)
                vis3d.points(points_world, color=(0,1,0), subsample=10, random=True, scale=0.001)
                vis3d.pose(T_camera_world)
                vis3d.pose(T_gripper_world_lift)
                vis3d.pose(T_gripper_world)
                vis3d.pose(T_cb_world)
                vis3d.pose(RigidTransform())
                vis3d.table(dim=0.5, T_table_world=T_cb_world)
                vis3d.show()

            if robot_type == 'ur5':
                robot.movej(KINEMATIC_AVOIDANCE_JOINTS, wait_for_res=True)
                robot.goto_pose(T_gripper_world_lift)
            else:
                robot.goto_pose(T_gripper_world_lift)
            robot.goto_pose(T_gripper_world)
            
            # wait for human measurement
            yesno = raw_input('Take measurement. Hit [ENTER] when done')
            robot.goto_pose(T_gripper_world_lift)

            # choose target point 2
            target_ind = np.where(ip == np.min(ip))[0]
            target_pt_world = cb_points_world[target_ind[0]]
                
            # create robot pose relative to target point
            t_gripper_world = np.array([target_pt_world.x + config['gripper_offset_x'],
                                        target_pt_world.y + config['gripper_offset_y'],
                                        target_pt_world.z + config['gripper_offset_z']])
            T_gripper_world = RigidTransform(rotation=R_gripper_world,
                                             translation=t_gripper_world,
                                             from_frame='gripper',
                                             to_frame='cb')
            logging.info('Moving robot to point x=%f, y=%f, z=%f' %(t_gripper_world[0], t_gripper_world[1], t_gripper_world[2]))
            
            T_lift = RigidTransform(translation=(0,0,0.05), from_frame='cb', to_frame='cb')
            T_gripper_world_lift = T_lift * T_gripper_world
            robot.goto_pose(T_gripper_world_lift)
            robot.goto_pose(T_gripper_world)
            
            # wait for human measurement
            yesno = raw_input('Take measurement. Hit [ENTER] when done')
            robot.goto_pose(T_gripper_world_lift)
            robot.goto_pose(T_orig_gripper_world_lift)

            # choose target point 3
            dir_world = np.array([1.0, 1.0, 0])
            dir_world = dir_world / np.linalg.norm(dir_world)
            ip = dir_world.dot(cb_point_data_world)
            target_ind = np.where(ip == np.max(ip))[0]
            target_pt_world = cb_points_world[target_ind[0]]
                
            # create robot pose relative to target point
            t_gripper_world = np.array([target_pt_world.x + config['gripper_offset_x'],
                                        target_pt_world.y + config['gripper_offset_y'],
                                        target_pt_world.z + config['gripper_offset_z']])
            T_gripper_world = RigidTransform(rotation=R_gripper_world,
                                             translation=t_gripper_world,
                                             from_frame='gripper',
                                             to_frame='cb')
            logging.info('Moving robot to point x=%f, y=%f, z=%f' %(t_gripper_world[0], t_gripper_world[1], t_gripper_world[2]))
            
            T_lift = RigidTransform(translation=(0,0,0.05), from_frame='cb', to_frame='cb')
            T_gripper_world_lift = T_lift * T_gripper_world
            robot.goto_pose(T_gripper_world_lift)
            robot.goto_pose(T_gripper_world)
            
            # wait for human measurement
            yesno = raw_input('Take measurement. Hit [ENTER] when done')
            robot.goto_pose(T_gripper_world_lift)
            robot.goto_pose(T_orig_gripper_world_lift)
            
            # stop robot
            robot.reset_home()
            if robot_type != 'ur5' and 'reset_bin' in config.keys() and config['reset_bin']:
                y.reset_bin()
            if robot_type == 'ur5':
                robot.stop()
            else:
                y.stop()
                
        sensor.stop()
            
