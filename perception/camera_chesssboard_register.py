import numpy as np
import logging
import cv2
import matplotlib.pyplot as plt
import math

from kinect2_sensor import Kinect2Sensor, Kinect2PacketPipelineMode
from image import DepthImage
from autolab_core import PointCloud, RigidTransform, rotation_from_axes, Point

class CameraChessboardRegister:

    _SENSOR_TYPES = ('kinect2',)

    @staticmethod
    def _find_chessboard(raw_image, sx=6, sy=9, vis=False):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((sx*sy,3), np.float32)
        objp[:,:2] = np.mgrid[0:sx,0:sy].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        # create images
        img = raw_image.data.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (sx,sy), None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            if False:
                cv2.drawChessboardCorners(img_rgb, (sx,sy), corners, ret)
                cv2.imshow('img', img_rgb)
                cv2.waitKey(500)
                #plt.figure()
                #plt.imshow(img_rgb)
                #plt.show()

            if corners is not None:
                return corners.squeeze()
        return None

    @staticmethod
    def get_camera_chessboard(sensor_frame, sensor_type, device_num, config):
        # load cfg
        num_transform_avg = config['num_transform_avg']
        num_images = config['num_images']
        sx = config['corners_x']
        sy = config['corners_y']
        color_image_upsample_rate = config['color_image_upsample_rate']
        vis = config['vis']

        # open sensor
        if sensor_type.lower() in CameraChessboardRegister._SENSOR_TYPES:
            sensor = Kinect2Sensor(device_num=device_num, frame=sensor_frame, packet_pipeline_mode=Kinect2PacketPipelineMode.CPU)
        else:
            logging.warning('Could not register device at %s. Sensor type %s not supported' %(sensor_frame, sensor_type))
        logging.info('Registering camera %s' %(sensor_frame))
        sensor.start()

        # repeat registration multiple times and average results
        R = np.zeros([3,3])
        t = np.zeros([3,1])
        points_3d_plane = PointCloud(np.zeros([3, sx*sy]), frame=sensor.ir_frame)

        k = 0
        while k < num_transform_avg:
            # average a bunch of depth images together
            depth_ims = np.zeros([Kinect2Sensor.DEPTH_IM_HEIGHT,
                                  Kinect2Sensor.DEPTH_IM_WIDTH,
                                  num_images])
            for i in range(num_images):
                small_color_im, new_depth_im, _ = sensor.frames()
                depth_ims[:,:,i] = new_depth_im.data

            med_depth_im = np.median(depth_ims, axis=2)
            depth_im = DepthImage(med_depth_im, sensor.ir_frame)

            # find the corner pixels in an upsampled version of the color image
            big_color_im = small_color_im.resize(color_image_upsample_rate)
            corner_px = CameraChessboardRegister._find_chessboard(big_color_im, sx=sx, sy=sy, vis=vis)

            if corner_px is None:
                logging.error('No chessboard detected')
                continue

            # convert back to original image
            small_corner_px = corner_px / color_image_upsample_rate

            if vis:
                plt.figure()
                plt.imshow(small_color_im.data)
                for i in range(sx):
                    plt.scatter(small_corner_px[i,0], small_corner_px[i,1], s=25, c='b')
                plt.axis('off')
                plt.show()

            # project points into 3D
            camera_intr = sensor.ir_intrinsics
            points_3d = camera_intr.deproject(depth_im)

            # get round chessboard ind
            corner_px_round = np.round(small_corner_px).astype(np.uint16)
            corner_ind = depth_im.ij_to_linear(corner_px_round[:,0], corner_px_round[:,1])
            if corner_ind.shape[0] != sx*sy:
                print 'Did not find all corners. Discarding...'
                continue

            # average 3d points
            points_3d_plane = (k * points_3d_plane + points_3d[corner_ind]) / (k + 1)
            logging.info('Registration iteration %d of %d' %(k+1, config['num_transform_avg']))
            k += 1

        # fit a plane to the chessboard corners
        X = np.c_[points_3d_plane.x_coords, points_3d_plane.y_coords, np.ones(points_3d_plane.num_points)]
        y = points_3d_plane.z_coords
        A = X.T.dot(X)
        b = X.T.dot(y)
        w = np.linalg.inv(A).dot(b)
        n = np.array([w[0], w[1], -1])
        n = n / np.linalg.norm(n)
        mean_point_plane = points_3d_plane.mean()

        # find x-axis of the chessboard coordinates on the fitted plane
        T_camera_table = RigidTransform(translation = -points_3d_plane.mean().data,
                                    from_frame=points_3d_plane.frame,
                                    to_frame='table')
        points_3d_centered = T_camera_table * points_3d_plane

        # get points along y
        coord_pos_y = int(math.floor(sx*(sy-1)/2.0))
        coord_neg_y = int(math.ceil(sx*(sy+1)/2.0))
        points_pos_y = points_3d_centered[:coord_pos_y]
        points_neg_y = points_3d_centered[coord_neg_y:]
        y_axis = np.mean(points_pos_y.data, axis=1) - np.mean(points_neg_y.data, axis=1)
        y_axis = y_axis - np.vdot(y_axis, n)*n
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(-n, y_axis)

        # WARNING! May need symmetry breaking but it appears the points are ordered consistently

        # produce translation and rotation from plane center and chessboard basis
        rotation_cb_camera = rotation_from_axes(x_axis, y_axis, n)
        translation_cb_camera = mean_point_plane.data
        T_cb_camera = RigidTransform(rotation=rotation_cb_camera,
                                     translation=translation_cb_camera,
                                     from_frame='cb',
                                     to_frame=sensor.frame)

        T_camera_cb = T_cb_camera.inverse()
        cb_points_cam = points_3d[corner_ind]

        # optionally display cb corners with detected pose in 3d space
        if config['debug']:
            # display image with axes overlayed
            cb_center_im = camera_intr.project(Point(T_cb_camera.translation, frame=sensor.ir_frame))
            cb_x_im = camera_intr.project(Point(T_cb_camera.translation + T_cb_camera.x_axis * config['scale_amt'], frame=sensor.ir_frame))
            cb_y_im = camera_intr.project(Point(T_cb_camera.translation + T_cb_camera.y_axis * config['scale_amt'], frame=sensor.ir_frame))
            cb_z_im = camera_intr.project(Point(T_cb_camera.translation + T_cb_camera.z_axis * config['scale_amt'], frame=sensor.ir_frame))
            x_line = np.array([cb_center_im.data, cb_x_im.data])
            y_line = np.array([cb_center_im.data, cb_y_im.data])
            z_line = np.array([cb_center_im.data, cb_z_im.data])

            plt.figure()
            plt.imshow(small_color_im.data)
            plt.scatter(cb_center_im.data[0], cb_center_im.data[1])
            plt.plot(x_line[:,0], x_line[:,1], c='r', linewidth=3)
            plt.plot(y_line[:,0], y_line[:,1], c='g', linewidth=3)
            plt.plot(z_line[:,0], z_line[:,1], c='b', linewidth=3)
            plt.axis('off')
            plt.title('Chessboard frame in camera %s' %(sensor.frame))
            plt.show()

        return T_camera_cb, cb_points_cam, points_3d_plane
