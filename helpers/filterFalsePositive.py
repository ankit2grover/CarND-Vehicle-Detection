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
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from helpers.car_positions import CarPositions


## Filter false positives through heatmap and filtering heats that have value lesser than threshold.
class FilterFalsePositive:
    
    def __init__(self):
        self.cars = None
        self.cars_count = 0
    
    ## Convert the image into heat(i.e. image with a black background)
    def convert_img_to_heat(self, image):
        self.heat = np.zeros_like(image[:,:,0]).astype(np.float)
        return self.heat
        
    ## Brighten the boxes region and overlapping boxes region will look like a heat on the black background image.
    def add_heat(self, boxes):
        for box in boxes:
            self.heat[box[0][1]: box[1][1], box[0][0]:box[1][0]] += 1
        return self.heat
    
    ## Filter the heat objects that are false positives(i.e. false positives heat value will be less than threshold)
    def threshold(self, threshold = 3):
        self.heat[self.heat < threshold] = 0
        return self.heat
    
    ## Get number of features in the image(i.e. number of cars)
    def labels_draw(self):
        self.labels = label(self.heat)
        cars_count = self.labels[1] + 1
        if ((self.cars_count is None) or self.cars_count == 0 or self.cars_count != cars_count):
            self.cars_count = cars_count
            self.cars = []
            for car_number in range(self.cars_count):
                self.cars.append(CarPositions())
        
    
    ## Draw Labeled boxes on the original image. Labelled boxes are calculated using labels() function.
    def draw_labeled_bboxes(self, image):
        img = np.copy(image)
        car_positions = CarPositions()
        for car_number in range(1, self.labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            nonzeroy = nonzero[0]
            nonzerox = nonzero[1]
            box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            print ("---Previous box---")
            print (box)
            tuple_box = (box[0][0], box[0][1], box[1][0], box[1][1])
            average_box_tuple = self.cars[car_number-1].update(tuple_box)
            box = ((average_box_tuple[0], average_box_tuple[1]), (average_box_tuple[2], average_box_tuple[3]))
            print ("---Average box---")
            print (box)
            img = cv2.rectangle(img, box[0], box[1], (0,0,255), 6)
        return img