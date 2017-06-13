import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class PerspectiveTransform :
    def __init__(self):
        self._M = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        self._Minv = cv2.getPerspectiveTransform(self.destination_points, self.source_points)
        
    @property
    def source_points(self):
        return np.array([(230, 700), (1075, 700), (693, 455), (588, 455)], np.float32)
    
    @property
    def destination_points(self):
        offset_left = 300
        offset_right = 300
        img_shape = (720, 1280)
        img_mid = img_shape[1]/2
        x1 = img_mid - offset_left
        x2 = img_mid + offset_right
        return np.array([(x1, img_shape[0]), (x2, img_shape[0]), (x2, 0), (x1, 0)], np.float32)
    
    @property
    def M(self):
        return self._M

    @property
    def Minv(self):
        return self._Minv
    
    def warp(self, image, newdims = None):
        if (newdims is None):
            newdims = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M, newdims, flags=cv2.INTER_LINEAR)
        
    def unwarp(self, image, newdims=None):
        if newdims is None:
            newdims = image.shape[1], image.shape[0]
        return cv2.warpPerspective(image, self.Minv, newdims, flags=cv2.INTER_LINEAR)

    