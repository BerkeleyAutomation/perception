import argparse
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from autolab_core import CameraIntrinsics
from perception import RgbdSensorFactory

parser = argparse.ArgumentParser("Camera calibration")
parser.add_argument(
    "chessboard_size", type=int, nargs=2, help="number of chessboard corners"
)
parser.add_argument(
    "calib_folder", type=str, help="path to folder of calibration images"
)
parser.add_argument(
    "--device", type=int, help="camera device number, if taking images"
)
parser.add_argument("--vis", type=bool, help="show images")
args = parser.parse_args()
chessboard_size = args.chessboard_size
calib_folder = args.calib_folder
device = args.device

# termination criteria and prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[
    0 : chessboard_size[0], 0 : chessboard_size[1]
].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = []
if calib_folder is not None and os.path.exists(calib_folder):
    images = glob.glob(os.path.join(calib_folder, "*.jpg"))
elif not os.path.exists(calib_folder):
    os.makedirs(calib_folder)

if len(images) == 0:
    # instantiate webcam sensor
    cam = RgbdSensorFactory.sensor(
        "webcam", {"frame": "webcam", "intrinsics": None, "device_id": device}
    )
    cam.start()
    for i in range(15):
        print("Press space key when chessboard is in view")
        while True:
            im = cam.frames()[0]
            cv2.imshow("img", im.data)
            if cv2.waitKey(1) == 32:
                break

        im_fname = os.path.join(calib_folder, f"calib_im_{i:02d}.jpg")
        im.save(im_fname)
        images.append(im_fname)
    cam.stop()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners)

        # Draw and display the corners
        if args.vis:
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow(fname, img)
            cv2.waitKey(2000)
if args.vis:
    cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
ci = CameraIntrinsics(
    "webcam",
    mtx[0, 0],
    mtx[1, 1],
    mtx[0, 2],
    mtx[1, 2],
    mtx[0, 1],
    *gray.shape[:2],
)
ci.save(os.path.join(calib_folder, "webcam.intr"))
np.save(os.path.join(calib_folder, "webcam_distortion.npy"), dist)

# Get reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print(f"Total reprojection error: {mean_error/len(objpoints):.4f}")
