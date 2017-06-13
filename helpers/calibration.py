import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class Camera:
    def __init__(self):
        self.mtx = None
        self.ret = None
        self.dist = None
        
    def chessboard_corners(self, img, dims=(9,6)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, dims, None)
        return ret, corners

    def calibrate(self, images, dims=(9,6)):
        img_points = []
        for image in images:
            ret, corners = self.chessboard_corners(image)
            if (ret):
                img_points.append(corners)
        
        objP = np.zeros((np.prod(dims), 3), np.float32)
        objP[:,:2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1,2)
        obj_points = [objP] * len(img_points)
        if (len(img_points) > 0):
            img_shape = images[0].shape[:2]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)
            if (ret):
                self.ret = ret
                self.mtx = mtx
                self.dist = dist
                return mtx, dist
        return None, None
    
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        