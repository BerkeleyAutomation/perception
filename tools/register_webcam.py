"""
Script to register webcam to a chessboard in the YuMi setup.
Authors: Matt Matl and Jeff Mahler
""" 
import argparse
import cv2
import logging
import numpy as np
import os
import traceback

from autolab_core import RigidTransform, YamlConfig
from perception import RgbdSensorFactory

from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Register a webcam to the robot')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/register_webcam.yaml', help='filename of a YAML configuration for registration')
    args = parser.parse_args()
    config_filename = args.config_filename
    config = YamlConfig(config_filename)

    T_cb_world = RigidTransform.load(config['chessboard_tf'])

    # Get camera sensor object
    for sensor_frame, sensor_data in config['sensors'].iteritems():
        logging.info('Registering {}'.format(sensor_frame))
        sensor_config = sensor_data['sensor_config']
        reg_cfg = sensor_data['registration_config'].copy()
        reg_cfg.update(config['chessboard_registration'])

        try:
            # Open sensor
            sensor_type = sensor_config['type']
            sensor_config['frame'] = sensor_frame
            logging.info('Creating sensor')
            sensor = RgbdSensorFactory.sensor(sensor_type, sensor_config)
            logging.info('Starting sensor')
            sensor.start()
            intrinsics = sensor.color_intrinsics
            logging.info('Sensor initialized')

            # Register sensor
            resize_factor = reg_cfg['color_image_rescale_factor']
            nx, ny = reg_cfg['corners_x'], reg_cfg['corners_y']
            sx, sy = reg_cfg['size_x'], reg_cfg['size_y']

            img, _, _= sensor.frames()
            resized_color_im = img.resize(resize_factor)
            corner_px = resized_color_im.find_chessboard(sx=nx, sy=ny)
            if corner_px is None:
                logging.error('No chessboard detected in sensor {}! Check camera exposure settings'.format(sensor_frame))
                exit(1)
            webcam_corner_px = corner_px / resize_factor

            # Compute Camera Matrix for webcam
            objp = np.zeros((nx*ny,3), np.float32)
            xstart = -sx * (nx / 2 - ((nx + 1) % 2) / 2.0)
            xend = sx * (nx / 2 - ((nx + 1) % 2) / 2.0 + 1)
            ystart = -sy * (ny / 2 - ((ny + 1) % 2) / 2.0)
            yend = sy * (ny / 2 - ((ny + 1) % 2) / 2.0 + 1)
            filler = np.mgrid[ystart:yend:sy, xstart:xend:sx]
            filler = filler.reshape((filler.shape[0], filler.shape[1] * filler.shape[2])).T
            objp[:,:2] = filler

            ret, rvec, tvec = cv2.solvePnP(objp, webcam_corner_px, intrinsics.K, None)
            mat, _ = cv2.Rodrigues(rvec)
            T_cb_cam = RigidTransform(mat, tvec, from_frame='cb', to_frame=sensor_frame)
            T_cam_cb = T_cb_cam.inverse()
            T_camera_world = T_cb_world.dot(T_cam_cb)

            logging.info('Final Result for sensor %s' %(sensor_frame))
            logging.info('Translation: ')
            logging.info(T_camera_world.translation)
            logging.info('Rotation: ')
            logging.info(T_camera_world.rotation)

        except Exception as e:
            logging.error('Failed to register sensor {}'.format(sensor_frame))
            traceback.print_exc()
            continue

        # save tranformation arrays based on setup
        output_dir = os.path.join(config['calib_dir'], sensor_frame)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pose_filename = os.path.join(output_dir, '%s_to_world.tf' %(sensor_frame))
        T_camera_world.save(pose_filename)
        intr_filename = os.path.join(output_dir, '%s.intr' %(sensor_frame))
        intrinsics.save(intr_filename)

        sensor.stop()
