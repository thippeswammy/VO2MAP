import os
import pickle

# import matplotlib.pyplot as plt
import cv2
import numpy as np

# # chessboard photos calibration constants
NX, NY = (7, 7)
Cal_img_path = r'DashCameraImg'

path_list = []
for file_name in os.listdir(Cal_img_path):
    s = Cal_img_path + "\\" + file_name
    path_list.append(s)


def get_camera_matrices():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((NX * NY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objPoints = []  # 3d points in real world space
    imgPoints = []  # 2d points in image plane.

    # Make a list of calibration images
    # Step through the list and search for chessboard corners
    img_size = tuple()
    for idx, fName in enumerate(path_list):
        curr_image = cv2.imread(fName)
        gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        # Assuming gray is your grayscale image and NX, NY are the number of inner corners in the chessboard pattern
        # cv2.imshow("sample", gray)
        # cv2.waitKey(1)

        ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

        # If found, add object points, image points
        if ret:  # The ret variable will indicate whether corners are found (True) or not (False)
            objPoints.append(objp)
            imgPoints.append(corners)
            # Optionally, display the image with corners
            cv2.drawChessboardCorners(curr_image, (NX, NY), corners, ret)
            cv2.imshow('Chessboard Corners', curr_image)
            cv2.waitKey(1)
        img_size = (gray.shape[1], gray.shape[0])
    cv2.destroyAllWindows()

    # Do camera calibration given object points and image points
    # ret: the return value indicating if the calibration was successful
    # mtx: the camera matrix containing intrinsic parameters
    # dist: the distortion coefficients
    # rVecs: the rotation vectors
    # tVecs: the translation vectors

    ret, mtx, dist, rVecs, tVecs = cv2.calibrateCamera(objPoints, imgPoints, img_size, None, None)

    if ret:
        data_camera = {'mtx': mtx, 'dist': dist}
        with open(Cal_img_path + 'CalibrationMatrix.p', 'wb') as f:
            pickle.dump(data_camera, f)
        print('created the camera matrices => done')
        print('saved at:', Cal_img_path + 'CalibrationMatrix.p')
        return mtx, dist, rVecs, tVecs
    else:
        raise "ERROR while calibration"


# Loading and saving the Camera_Matrices;
def loadCameraMatrices(camera_file_path=Cal_img_path + 'CalibrationMatrix.p'):
    print(os.listdir('.'))
    if camera_file_path not in os.listdir('.'):
        mtx, dist, _, _ = get_camera_matrices()
    else:
        camera_file = open(camera_file_path, 'rb')
        cameraMatrices = pickle.load(camera_file)
        mtx, dist = cameraMatrices['mtx'], cameraMatrices['dist']
    return mtx, dist


print(loadCameraMatrices())
