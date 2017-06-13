import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from .line import Line

class LaneSearch:
    
    def __init__(self):
        self._draw_image = None
        self._image = None
        self._left_fit_coeff = None
        self._right_fit_coeff = None
        self.left_line = Line()
        self.right_line = Line()
        self.confident = False
        
    @property
    def draw_image(self):
        return self._draw_image
    
    @property
    def image(self):
        return self._image
    
    @property
    def left_fit_coeff(self):
        return self._left_fit_coeff
    
    @property
    def right_fit_coeff(self):
        return self._right_fit_coeff
    
    
    def sliding_window(self, warp_image):
        self._draw_image = warp_image
        self._image = warp_image
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        hist = np.sum(self.image[self.image.shape[0]//2:,:], axis = 0)
        midpoint = np.int(hist.shape[0]/2)
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint
        ## Create an output image to visualize the result.
        img_frame = self.image
        out_img = np.dstack((img_frame, img_frame, img_frame)) * 255
        ## Number of windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img_frame.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            win_y_low = img_frame.shape[0] - ((window + 1) * window_height)
            win_y_high = img_frame.shape[0] - (window  * window_height)
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0),2)
            good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each lane and get their coefficients.
        self._left_fit_coeff = np.polyfit(lefty, leftx, 2)
        self._right_fit_coeff = np.polyfit(righty, rightx, 2)
        self._draw_image = out_img
        self.confident = True
        return self._draw_image
    
        
    def sliding_window_look_ahead_filter(self, warp_image):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = warp_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self._left_fit_coeff[0]*(nonzeroy**2) + self._left_fit_coeff[1]*nonzeroy + self._left_fit_coeff[2] - margin)) & (nonzerox < (self._left_fit_coeff[0]*(nonzeroy**2) + self._left_fit_coeff[1]*nonzeroy + self._left_fit_coeff[2] + margin))) 
        right_lane_inds = ((nonzerox > (self._right_fit_coeff[0]*(nonzeroy**2) + self._right_fit_coeff[1]*nonzeroy + self._right_fit_coeff[2] - margin)) & (nonzerox < (self._right_fit_coeff[0]*(nonzeroy**2) + self._right_fit_coeff[1]*nonzeroy + self._right_fit_coeff[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self._left_fit_coeff = np.polyfit(lefty, leftx, 2)
        self._right_fit_coeff = np.polyfit(righty, rightx, 2)
        
    def draw_lane_results(self, warp_image, Minv, undist_img):
        ## Visualization
        # Generate x and y values for plotting
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warp_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        ploty = np.linspace(0, self._draw_image.shape[0] -1, self._draw_image.shape[0])
        left_fitx = self._left_fit_coeff[0]*ploty**2 + self._left_fit_coeff[1]*ploty + self._left_fit_coeff[2]
        right_fitx = self._right_fit_coeff[0]*ploty**2 + self._right_fit_coeff[1]*ploty + self._right_fit_coeff[2]
        # Calcualate radius of curvature on the image frame.
        curv_radius = self._calc_radius_curvature_world_space(ploty, left_fitx, right_fitx)
        ## Average value of radius of curvature of left and right lanes
        curv = (curv_radius[0] + curv_radius[1]) / 2
        ## Perform sanity check.
        accepted = self._sanity_check(self.left_line.radius_of_curvature, curv)
        ## Process the results and if accepted do average of fits, average coefficients, else returns previous average results.
        self.left_line.process(accepted, left_fitx, self._left_fit_coeff, curv)
        self.right_line.process(accepted, right_fitx, self._right_fit_coeff, curv)
        if (accepted):
            ## Update the coefficients with average values
            self._left_fit_coeff = self.left_line.best_fit
            self._right_fit_coeff = self.right_line.best_fit
            ## Update the fits x values as per average coefficeints.
            left_fitx = self._left_fit_coeff[0]*ploty**2 + self._left_fit_coeff[1]*ploty + self._left_fit_coeff[2]
            right_fitx = self._right_fit_coeff[0]*ploty**2 + self._right_fit_coeff[1]*ploty + self._right_fit_coeff[2]
        
        #print (left_fitx)
        pts_left = np.array([np.transpose(np.vstack((left_fitx, ploty)))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack((right_fitx, ploty))))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255,0))
        ## Calculate x offset and show the offset on the image
        offset = self.calc_offset(self._left_fit_coeff, self._right_fit_coeff)
        offset_text = "Offset is {:.2f}m".format(abs(offset))
        if (offset < 0):
            offset_text += " left of center"
        else :
            offset_text += " right of center"
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwarp_img_with_results = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
        undist_img_with_results = cv2.addWeighted(undist_img, 1, unwarp_img_with_results, 0.3, 0)
        ## Show curvature of radius on the image frame.
        curve_text = "Average radius of curvature is {:.2f}m".format(curv)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undist_img_with_results, curve_text, (320, 60), font, 1, (255, 255, 255), 2)
        cv2.putText(undist_img_with_results, offset_text, (320, 120), font, 1, (255, 255, 255), 2)
        return undist_img_with_results 
    
    def _calc_radius_curvature_pixels(self, y):
        left_curve_radius = ((1 + (2*self._left_fit_coeff[0]*y + self._left_fit_coeff[1])**2)**(1.5))/ np.absolute(2*2*self._left_fit_coeff[0]) 
        right_curve_radius = ((1 + (2*self._right_fit_coeff[0]*y + self._right_fit_coeff[1])**2)**(1.5))/ np.absolute(2*2*self._right_fit_coeff[0]) 
        return (left_curve_radius, right_curve_radius)
    
    def _calc_radius_curvature_world_space(self, ploty, left_fitx, right_fitx):
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3/600 # meters per pixel in x dimension
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx* xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx* xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        return (left_curverad, right_curverad)
    
    def calc_offset(self, left_coeff, right_coeff, y = 720):
        left_x = left_coeff[0]*y**2 + left_coeff[1]*y + left_coeff[2]
        right_x = right_coeff[0]*y**2 + right_coeff[1]*y + right_coeff[2]
        center_x = (left_x + right_x) /2
        offset = center_x - 640
        xm_per_pix = 3/600 # meters per pixel in x dimension
        offset = offset * xm_per_pix
        return offset
        
    def _sanity_check(self, prev_curv_radius, curv_radius, thresh = 0.5):
        ### Returns true if prvious curvature radius is none as it is first entry.
        if (prev_curv_radius is None):
            return False
        ### Returns true if change in radius of curvature is less than 50% else returns false.
        diff_radius = curv_radius - prev_curv_radius
        change = diff_radius / curv_radius
        change = (np.absolute(change))
        return (change <= thresh)
        