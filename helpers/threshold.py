import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class ThresholdUtil:
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        abs_sobel = None
        if (orient == 'x'):
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
        elif (orient == 'y'):
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    # Define a function that applies Sobel x and y, 
    # then computes the magnitude of the gradient
    # and applies a threshold
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Calculate the magnitude 
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        # 5) Create a binary mask where mag thresholds are met
        # 6) Return this mask as your binary_output image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        abs_sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        abs_sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_sobel = np.absolute(np.sqrt(abs_sobelx ** 2 + abs_sobely ** 2))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        return sxbinary

    # Define a function that applies Sobel x and y, 
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):

        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        # 5) Create a binary mask where direction thresholds are met
        # 6) Return this mask as your binary_output image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
        abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
        #abs_sobel = np.absolute(np.sqrt(abs_sobelx ** 2 + abs_sobely ** 2))
        sobel_arctan2 = np.arctan2(abs_sobely, abs_sobelx)
        sxbinary = np.zeros_like(sobel_arctan2)
        sxbinary[(sobel_arctan2 >= thresh[0]) & (sobel_arctan2 <= thresh[1])] = 1
        return sxbinary
    
    def create_combined_thresh(self, image):
        # Choose a Sobel kernel size
        ksize = 15
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=29, thresh=(20, 255))
        grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=15, thresh=(30, 255))
        mag_binary = self.mag_thresh(image, sobel_kernel=15, mag_thresh=(30, 250))
        dir_binary = self.dir_threshold(image, sobel_kernel=ksize, thresh=(0.2, 1.5))

        combined = np.zeros_like(dir_binary)
        combined[((gradx ==1) & (grady == 1)) & ((dir_binary == 1) & (mag_binary == 1))] = 1
        ##combined[((gradx ==1))  & ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined
    
    def create_combined_color_thresh(self, image):
        # Convert to HLS color space and separate the V channel
        s_thresh=(170, 255)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        r_channel = image[:,:,0]
        g_channel = image[:,:,1]
        b_channel = image[:,:,2]

        sobel_combined_binary = self.create_combined_thresh(image)

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        # color_binary = np.dstack((np.zeros_like(sobel_combined_binary), sobel_combined_binary, s_binary))
        
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sobel_combined_binary)
        ## Get gray map image containing directional and color thresholds only
        combined_binary[(sobel_combined_binary == 1) | (s_binary == 1)] = 1
        return combined_binary