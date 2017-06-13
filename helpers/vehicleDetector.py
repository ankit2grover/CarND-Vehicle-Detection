import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import numpy as np
import glob
from skimage.feature import hog

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
from skimage.feature import hog
from skimage import data, color, exposure
from .utils import *
from collections import deque

'''
Class to detect cars on the image frame using sliding window hog subsampling approach
HOG Subsampling approach computes HOG features on the sliding windows and then subsamples
the images into different boxes to compute spatial and histogram features also.
Henceforth it will use all the three features to classify the subsampled image is Car or not.
'''
class VehicleDetector:

	def __init__(self):
		self.xstart = 600
		# Various Scales
		self.ystart_ystop_scales_list = [(360, 560, 1.5), (400, 600, 1.8), (440, 700, 2.5)]
		self.utils = Utils()
		self.xstart = 400

	# Define a single function that can extract features using hog sub-sampling and make predictions
	def find_cars(self, img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, window = 64):
		img = np.copy(img)
		img = img.astype(np.float32)/255
		bboxes_list = []

		for (ystart, ystop, scale) in self.ystart_ystop_scales_list:
			img_tosearch = img[ystart:ystop, self.xstart: ,:]
			ctrans_tosearch = self.utils.convert_color(img_tosearch, color_space ='YCrCb')
			if scale != 1:
				imshape = ctrans_tosearch.shape
				ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

			ch1 = ctrans_tosearch[:,:,0]
			ch2 = ctrans_tosearch[:,:,1]
			ch3 = ctrans_tosearch[:,:,2]
			# Define blocks and steps as above
			nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
			nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
			nfeat_per_block = orient*cell_per_block**2
			# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
			# window = 64
			nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
			cells_per_step = 2  # Instead of overlap, define how many cells to step
			nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
			nysteps = (nyblocks - nblocks_per_window) // cells_per_step
			# Compute individual channel HOG features for the entire image
			hog1 = self.utils.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
			hog2 = self.utils.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
			hog3 = self.utils.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
			bboxes = []
			for xb in range(nxsteps):
				for yb in range(nysteps):
					ypos = yb*cells_per_step
					xpos = xb*cells_per_step
					# Extract HOG for this patch
					hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
					hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
					hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
					hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
					xleft = xpos*pix_per_cell
					ytop = ypos*pix_per_cell
					# Extract the image patch
					subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (window, window))
					#subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
					# Get color features
					spatial_features = self.utils.bin_spatial(subimg, size=spatial_size)
					hist_features = self.utils.color_hist(subimg, nbins=hist_bins)
					# Scale features and make a prediction
					test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
					#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
					test_prediction = svc.predict(test_features)
					if test_prediction == 1:
						xbox_left = np.int(xleft*scale) + self.xstart
						ytop_draw = np.int(ytop*scale)
						win_draw = np.int(window*scale)
						box = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
						bboxes_list.append(box)

		for box in bboxes_list:	      
			cv2.rectangle(img, box[0], box[1], (0,0,255),6) 

		return img, bboxes_list