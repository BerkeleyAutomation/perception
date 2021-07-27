"""
Script to register webcam to a chessboard in the YuMi setup.
Authors: Matt Matl and Jeff Mahler
"""
import argparse
import logging
import os
import traceback

import cv2
import numpy as np
from autolab_core import RigidTransform, YamlConfig

from perception import RgbdSensorFactory

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(
        description="Register a webcam to the robot"
    )
    parser.add_argument(
        "--config_filename",
        type=str,
        default="cfg/tools/register_webcam.yaml",
        help="filename of a YAML configuration for registration",
    )
    args = parser.parse_args()
    config_filename = args.config_filename
    config = YamlConfig(config_filename)
    reg_cfg = config["chessboard"]
    sensor_cfg = config["sensor"]

    T_cb_world = RigidTransform.load(config["chessboard"]["tf"])

    # Get camera sensor object
    sensor_frame = sensor_cfg["frame"]
    logging.info("Registering {}".format(sensor_frame))
    try:
        # Open sensor
        logging.info("Creating sensor")
        sensor = RgbdSensorFactory.sensor(sensor_frame, {"frame": sensor_frame, "intrinsics": sensor_cfg["intrinsics"], "device_id": sensor_cfg["device_id"]})
        logging.info("Starting sensor")
        sensor.start()
        intrinsics = sensor.color_intrinsics
        logging.info("Sensor initialized")

        # Register sensor
        nx, ny = reg_cfg["corners_x"], reg_cfg["corners_y"]
        sx, sy = reg_cfg["size_x"], reg_cfg["size_y"]

        img, _ = sensor.frames()
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # create images and find the chess board corners
        gray = cv2.cvtColor(img.data, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, (nx, ny), None)
        if not ret:
            logging.error(
                "No chessboard detected in sensor {}! "
                "Check camera exposure settings".format(sensor_frame)
            )
            exit(1)

        # Compute Camera Matrix for webcam
        objp = np.zeros((nx * ny, 3), np.float32)
        xstart = -sx * (nx / 2 - ((nx + 1) % 2) / 2.0)
        xend = sx * (nx / 2 - ((nx + 1) % 2) / 2.0)
        ystart = -sy * (ny / 2 - ((ny + 1) % 2) / 2.0)
        yend = sy * (ny / 2 - ((ny + 1) % 2) / 2.0)
        filler = np.mgrid[ystart:yend:sy, xstart:xend:sx]
        filler = filler.reshape(
            (filler.shape[0], filler.shape[1] * filler.shape[2])
        ).T
        objp[:, :2] = filler

        ret, rvec, tvec = cv2.solvePnP(
            objp, corners.squeeze(), intrinsics.K, None
        )
        mat, _ = cv2.Rodrigues(rvec)
        T_cb_cam = RigidTransform(
            mat, tvec, from_frame="cb", to_frame=sensor_frame
        )
        T_cam_cb = T_cb_cam.inverse()
        T_camera_world = T_cb_world.dot(T_cam_cb)

        logging.info("Final Result for sensor %s" % (sensor_frame))
        logging.info("Translation: ")
        logging.info(T_camera_world.translation)
        logging.info("Rotation: ")
        logging.info(T_camera_world.rotation)

    except Exception:
        logging.error("Failed to register sensor {}".format(sensor_frame))
        traceback.print_exc()

    # save tranformation arrays based on setup
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pose_filename = os.path.join(
        output_dir, "%s_to_world.tf" % (sensor_frame)
    )
    T_camera_world.save(pose_filename)
    sensor.stop()
