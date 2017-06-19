"""
Class for Registring a WebCam using a chessboard and
the conversion to go from 2D robot pose (x,y) to 2D image pixel (x,y).
Author: Michael Laskey

"""

import logging, cv2, os
import numpy as np
from numpy import linalg as LA
import scipy.spatial.distance as ssd
import scipy.optimize as opt
from autolab_core import RigidTransform, YamlConfig
from perception import bincam_2D as b2


class RegWC():
    def __init__(self, cfg, cid, options=None):
        self.cfg = cfg

        self.trans = None
        self.bc = b2.BinaryCamera()
        self.bc.open(cid)

        #Camera Parameters
        f_x = 790
        f_y = 790
        c_x = 240
        c_y = 320

        self.C = np.array([[f_x,0.0,c_x],[0.0,f_y,c_y],[0.0,0.0,1.0]])
        self.p_x_off = 0
        self.p_y_off = 0
        if(not options == None ):
            self.p_x_off = options.OFFSET_X
            self.p_y_off = options.OFFSET_Y
            self.height = options.HEIGHT
            self.width = options.WIDTH

        #Chessboard Facts (M)
        self.W = self.cfg['chessboard_registration']['width']
        self.H = self.cfg['chessboard_registration']['height']
        self.Row= self.cfg['chessboard_registration']['corners_x']
        self.Column = self.cfg['chessboard_registration']['corners_y']
        self.x_off = self.cfg['chessboard_registration']['x_off']
        self.y_off = self.cfg['chessboard_registration']['y_off']

    def get_image_bounds(self):
        """
        Get the robot bounds of the image specifed by an option file

        Returns:
        numpy array:
            (2,) shape array that is the (x,y) lower bound in robot pose

        numpy array
            (2,) shape array that is the (x,y) upper bound in robot pose

        """

        c1 = np.array([0,0])
        c2 = np.array([self.width,self.height])

        c1 = self.pixel_to_robot(c1)
        c2 = self.pixel_to_robot(c2)
        return c1,c2

    def offset_camera(self,pixel):
        """
        Accounts for the orign shift when using the options file

        Parameters
        ----------
        pixel: (2,) shape numpy array

        Returns
        -------
        numpy array:
            (2,) shape array that is the (x,y) pixel when shifted

        """
        pixel[0] = pixel[0]-self.p_x_off
        pixel[1] = pixel[1]-self.p_y_off

        return pixel

    def offset_cam_back(self,pixel):
        """
        Goes from shifted camera origin back to original camera orign

        Parameters
        ----------
        pixel: (2,) shape numpy array

        Returns
        -------
        numpy array:
            (2,) shape array that is the (x,y) pixel of the original camera

        """
        pixel[0] = pixel[0]+self.p_x_off
        pixel[1] = pixel[1]+self.p_y_off

        return pixel

    def pixel_to_robot(self,pixel):
        """
        Takes a point in pixel space and converts it to robot space

        Parameters
        ----------
        pixel: (2,) shape numpy array
            Pixel position in the image

        Returns
        -------
        numpy array:
            (2,) shape array that is the (x,y) pixel of the original camera

        """
        p_reg = np.zeros([3])
        p_reg[0:2] = self.offset_cam_back(pixel[0:2])
        p_reg[2] = 1

        #Pixel Ray
        ray = np.matmul(LA.inv(self.C),p_reg)
        ray = ray/LA.norm(ray)

        #Chessboard Plane
        z_axis = np.array([0,0,1])
        rot = self.trans[0:3,0:3]
        P_0 = self.trans[:,3]
        z_axis_rot = np.matmul(rot.T,-P_0)

        #d = np.matmul(z_axis_rot,P_0)
        d = np.matmul(rot.T,ray)
        p_prog = np.zeros(3)
        p_prog[0] = (-z_axis_rot[2]/d[2])*d[0]+z_axis_rot[0]
        p_prog[1] = (-z_axis_rot[2]/d[2])*d[1]+z_axis_rot[1]

        return self.chessboard_to_robot(p_prog)

    def chessboard_to_robot(self,cords):
        """
        Translates a point from cheesboard frame to robot

        Parameters
        ----------
        pixel: (2,) shape numpy array
            Position in the cheesboard fraem

        Returns
        -------
        numpy array:
            (2,) shape array that is the (x,y) pose in robot frame

        """
        #Translate to Robot Frame
        #+X in Chessboard is +X in Chessboard
        #+X in Robot is -Y in Chessboard
        c_r = np.zeros(2)
        c_r[0] = -cords[1]+self.x_off
        c_r[1] = cords[0] + self.y_off

        return c_r

    def robot_to_chessboard(self,cords):
        """
        Translates a point from robot frame to chessboard

        Parameters
        ----------
        pixel: (2,) shape numpy array
            (x,y) pose in robot frame

        Returns
        -------
        numpy array:
            (2,) shape array that is position in the cheesboard fraem

        """
        #Translate to Chessboard Frame
        #+Y in Robot is +X in Chessboard
        #+X in Robot is -Y in Chessboard
        temp = cords[0]

        cords[0] = cords[1] - self.y_off
        cords[1] = -(temp - self.x_off)
        c_z = np.zeros(4)
        c_z[0:2] = cords
        c_z[3] = 1.0
        return c_z

    def robot_to_pixel(self,cords):
        """
        Takes a point in robot space and converts it to pixel space

        Parameters
        ----------
        pixel: (2,) shape numpy array
            (x,y) Robot pose

        Returns
        -------
        numpy array:
            (2,) shape array that is the (x,y) pixel position

        """

        c_z = self.robot_to_chessboard(cords)

        ext = np.matmul(self.C,self.trans)
        p_un = np.matmul(ext,c_z)

        p = np.zeros(2)

        p[0] = p_un[0]/p_un[2]
        p[1] = p_un[1]/p_un[2]

        return self.offset_camera(p)

    def register_chessboard(self):
        """
        Compute the transformation between robot pose and camera pose using chessboard registration
        techniques

        Returns
        -------
        numpy array:
            (3,4) shape array that is the computed transformation

        """

        p_mm = self.get_corners_mm()
        c_mm = self.find_chessboard()

        dist_coef = np.zeros(4)

        ret,r_vec,t_vec = cv2.solvePnP(p_mm.T,c_mm.T,self.C,dist_coef)

        r_mat,j = cv2.Rodrigues(r_vec)

        trans = np.zeros([3,4])

        trans[0:3,0:3] = r_mat
        trans[:,3] = t_vec[:,0]

        self.trans = trans
        return trans

    def get_corners_mm(self):
        """
        Computes each mm position of a cheeseboard given an origin that is the center of
        the chessboard

        Returns
        -------
        numpy array:
            (3,54) shape array that is each corner in the chessboard

        """
        top_right = np.zeros(2)
        points = np.zeros([3,54],dtype=np.float32)

        #TOP RIGHT CORNER
        top_right[0] = int(self.Column/2)*self.W
        top_right[1] = self.H*0.5+self.H*int(self.Row/2-1)
        idx = 0
        for i in range(self.Column):
            for j in range(self.Row):
                point = np.copy(top_right)
                point[0] = point[0] - i*self.W
                point[1] = point[1] - j*self.H
                points[0:2,idx] = point
                idx += 1

        return points

    def find_chessboard(self,debug=False):
        """
        Finds the cheeseboard in the image and recovers each corner

        Parameters
        ----------
        debug: Boolean
            If True, shows image with cheeseboard corners overlayed (Defaul False)

        Returns
        -------
        numpy array:
            (2,54) shape array that is each corner in the chessboard

        """
        ic = np.array([self.Row,self.Column])
        img = self.bc.read_raw()
        ret,ic = cv2.findChessboardCorners(img,(6,9))

        ic_np = np.zeros([2,54])

        for i in range(self.Column*self.Row):
            ic_np[:,i] = ic[i][0,:]

        if(debug):
            for i in range(len(ic)):
                p = ic[i]
                img[int(p[0][1]),int(p[0][0]),2] = 255

            cv2.imshow('debug',img)
            cv2.waitKey(30)
            IPython.embed()

        return ic_np

    def robot_to_pixel_scale(self,scale):
        """
        Not Supported
        """
        base = np.zeros(2)
        measures = np.array([0,scale])

        base_p = self.robot_to_pixel(base)
        m_p = self.robot_to_pixel(measures)

        return LA.norm(m_p-base_p)

if __name__ == "__main__":
    """
    Run to register camera make sure cheesboard can be seen in image
    """
    logging.getLogger().setLevel(logging.INFO)

    cfg_filename = 'cfg/tools/register_webcam.yaml'
    cfg = YamlConfig(cfg_filename)

    for sensor_frame, sensor_config in cfg['sensors'].iteritems():
        if sensor_config['use']:
            logging.info("Registering {0}".format(sensor_frame))
            cid = sensor_config['device_num']
            reg = RegWC(cfg, cid)
            T_mat = reg.register_chessboard()
            T = RigidTransform(rotation=T_mat[:3,:3], translation=T_mat[:,3], from_frame='{0}'.format(sensor_frame), to_frame='world')
            output_dir = os.path.join(cfg['calib_dir'], sensor_frame)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_filename = os.path.join(output_dir, '{0}_to_world.tf'.format(sensor_frame))
            logging.info("Saving to {0}".format(output_filename))
            T.save(output_filename)
