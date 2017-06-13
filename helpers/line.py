import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, n_history= 60):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_nfitted = deque([], n_history) 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # Double ended queue containing n recent coefficients
        self.history_coeff = deque([], n_history);
        
    def process(self, accepted, x_fits, coeff, radius_of_curvature):
        self.recent_nfitted.append((accepted, x_fits))
        self.history_coeff.append((accepted, coeff))
        self.detected = accepted
        self.radius_of_curvature = radius_of_curvature
        self.diffs = self.diffs - np.array([coeff])
        self.diffs = self.diffs.astype(np.float32)
        if (accepted):
            self.current_fit = np.array(coeff)
            if (self.best_fit is None):
                self.best_fit = np.array(coeff)
            if (self.bestx is None):
                self.bestx = np.array(coeff)
            if (len(self.best_fit) >= 2):
                ## Calculate weighted average of the coefficients and fit leftxs and rightxs
                all_coeff = list(filter(lambda h: h[0], self.history_coeff))
                all_recent_fitted = list(filter(lambda fitted: fitted[0], self.recent_nfitted))
                total_num = len(all_coeff)
                all_coeff = np.array([h[1] for h in all_coeff[-total_num:]])
                weights = np.array([1 for i in range(total_num, 0, -1)])
                self.best_fit = np.average(all_coeff, weights= weights, axis = 0)
                self.bestx = np.average(np.array([h[1] for h in all_recent_fitted]), axis = 0)
        