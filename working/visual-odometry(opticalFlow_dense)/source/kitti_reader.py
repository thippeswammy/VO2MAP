import os
from math import sqrt

import cv2
import numpy as np


class DatasetReaderKITTI:
    def __init__(self, datasetPath, scaling=1.0):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        # self._datasetPath = os.path.join(__location__, datasetPath)
        self._datasetPath = datasetPath
        self._imagesPath = os.path.join(self._datasetPath, "image_1")
        # print('os.listdir(self._imagesPath)', os.listdir(self._imagesPath))
        self._numFrames = len([x for x in os.listdir(self._imagesPath) if x.endswith(".png")])
        self._scaling = scaling

        if self._numFrames < 2:
            raise Exception("Not enough images ({}) found, aborting.".format(frameReader.getFramesCount()))
        else:
            print("Found {} images in {}".format(self._numFrames, self._imagesPath))

    def readFrame(self, index=0):
        if index >= self._numFrames:
            raise Exception("Cannot read frame number {} from {}".format(index, self._imagesPath))

        img = cv2.imread(os.path.join(self._imagesPath, "{:06d}.png".format(index)))
        img = cv2.resize(img, (int(img.shape[1] * self._scaling), int(img.shape[0] * self._scaling)))
        return img

    def readCameraMatrix(self):
        cameraFile = os.path.join(self._datasetPath, "calib.txt")
        with open(cameraFile) as f:
            firstLine = f.readlines()[0][4:]
            focal, _, cx, _, _, _, cy, _, _, _, _, _ = list(map(float, firstLine.rstrip().split(" ")))

            K = np.zeros((3, 3))
            # K[0, 0] = focal
            # K[0, 2] = cx
            # K[1, 1] = focal
            # K[1, 2] = cy
            # K[2, 2] = 1
            K[0, 0] = 718.8560
            K[0, 2] = 607.1928
            K[1, 1] = 718.8560
            K[1, 2] = 185.2157
            K[2, 2] = 1
            return K

    def readGroundtuthPosition(self, frameId):
        groundtruthFile = os.path.join(self._datasetPath, "poses.txt")
        with open(groundtruthFile) as f:
            lines = f.readlines()

            x11, x12, x13, tx, y11, y12, y13, ty, z11, z12, z13, tz = list(
                map(float, lines[frameId].rstrip().split(" ")))
            x21, x22, x23, tx_prev, y21, y22, y23, ty_prev, z21, z22, z23, tz_prev = list(
                map(float, lines[frameId - 1].rstrip().split(" ")))

            position = (tx, ty, tz)
            scale = sqrt((tx - tx_prev) ** 2 + (ty - ty_prev) ** 2 + (tz - tz_prev) ** 2)

            return position, scale

    def readGroundTruthPositionRotation(self, frameId):
        groundtruthFile = os.path.join(self._datasetPath, "poses.txt")
        with open(groundtruthFile) as f:
            lines = f.readlines()

            x11, x12, x13, tx, y11, y12, y13, ty, z11, z12, z13, tz = list(
                map(float, lines[frameId].rstrip().split(" ")))
            x21, x22, x23, tx_prev, y21, y22, y23, ty_prev, z21, z22, z23, tz_prev = list(
                map(float, lines[frameId - 1].rstrip().split(" ")))

            position = (tx, ty, tz)
            scale = sqrt((tx - tx_prev) ** 2 + (ty - ty_prev) ** 2 + (tz - tz_prev) ** 2)

            R_curr = np.array([[x11, x12, x13],
                               [y11, y12, y13],
                               [z11, z12, z13]])

            R_prev = np.array([[x21, x22, x23],
                               [y21, y22, y23],
                               [z21, z22, z23]])

            return position, scale, R_curr, R_prev

    def getFramesCount(self):
        return self._numFrames

    def getDatasetPath(self):
        return self._datasetPath
