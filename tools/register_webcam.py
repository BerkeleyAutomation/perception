"""
Class for gettign Webcam Register
Author: Michael Laskey
"""

import copy
import IPython
import logging
import numpy as np
import cv2
import scipy.spatial.distance as ssd
import scipy.optimize as opt

#from alan.core import RigidTransform, PointCloud, NormalCloud
from perception import OpenCVCameraSensor
from numpy import linalg as LA

class RegWC():
    def __init__(self, id):
        self.X_MM = 200
        self.Y_MM = 400
        self.X_P = 10
        self.Y_P = 10

        self.trans = None
        self.cam = OpenCVCameraSensor(id)
        self.cam.start()

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
        self.W = 0.023
        self.H = 0.024
        self.Row= 6
        self.Column = 9
        self.x_off = 0.454
        self.y_off = 0.0107

        CEN_TO_BASE = 0.0

    def get_image_bounds(self):
        c1 = np.array([0,0])
        c2 = np.array([self.width,self.height])

        c1 = self.pixel_to_robot(c1)
        c2 = self.pixel_to_robot(c2)

        return c1,c2

    def offset_camera(self,pixel):
        pixel[0] = pixel[0]-self.p_x_off
        pixel[1] = pixel[1]-self.p_y_off

        return pixel

    def robot_to_pixel_scale(self,scale):
        base = np.zeros(2)
        measues = np.array([0,scale])

        base_p = self.robot_to_pixel(base)
        m_p = self.robot_to_pixel(measues)

        return LA.norm(m_p - base_p)

    def offset_cam_back(self,pixel):
        pixel[0] = pixel[0]+self.p_x_off
        pixel[1] = pixel[1]+self.p_y_off

        return pixel

    def robot_to_pixel_scale(self,scale):
        base = np.zeros(2)
        measures = np.array([0,scale])

        base_p = self.robot_to_pixel(base)
        m_p = self.robot_to_pixel(measures)

        return LA.norm(m_p-base_p)

    def pixel_to_robot(self,pixel):
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
        #Translate to Robot Frame
        #+X in Chessboard is +X in Chessboard
        #+X in Robot is -Y in Chessboard
        c_r = np.zeros(2)
        c_r[0] = -cords[1]+self.x_off
        c_r[1] = cords[0] + self.y_off

        return c_r

    def robot_to_chessboard(self,cords):
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
        c_z = self.robot_to_chessboard(cords)

        ext = np.matmul(self.C,self.trans)
        p_un = np.matmul(ext,c_z)

        p = np.zeros(2)
        #IPython.embed()

        p[0] = p_un[0]/p_un[2]
        p[1] = p_un[1]/p_un[2]

        return self.offset_camera(p)

    def register_chessboard(self):
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
        #Rows
        ic = np.array([self.Row,self.Column])

        img = self.bc.read_raw()

        ret,ic = cv2.findChessboardCorners(img,(6,9))

        ic_np = np.zeros([2,54])

        cv2.imshow('debug',img)
        cv2.waitKey(30)
        IPython.embed()

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

if __name__ == "__main__":

    reg = RegWC(calibrate=True)
    trans = reg.register_chessboard()
    IPython.embed()

    c_m = reg.find_chessboard()
    p_m = reg.get_corners_mm()
    top_right = p_m[:,0]

    print "mm ",reg.chessboard_to_robot(p_m[:,0])

    points = reg.pixel_to_robot(c_m[:,0])

    print points
    #Save Registration To File
    trans = reg.trans

    pickle.dump(trans,open('data/registration/registration.pckl','wb'))
