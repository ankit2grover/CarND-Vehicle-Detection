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
from .utils import Utils
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif
##Evaluate accuracy of your model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC

'''
Classifier Utility to perform Linear SVC classifier on car and non car images.
It classify the train data using Linear SVC and accuracy, precision and recall score is used to
compute the efficieny of classfier.
'''
class ClassifierUtil:

	
	def run_classifier(self, car_images_files, not_car_images_files):
		# Read in cars and notcars
		#print (glob.glob('*'))
		print ("Reading all the images")
		car_images = []
		##car_images_files = glob.glob('train_images/vehicles/**/*.png', recursive = True)
		##not_car_images_files = glob.glob('train_images/non-vehicles/**/*.png', recursive = True)
		print ("Car images files {}".format(len(car_images_files)))
		color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
		orient = 12  # HOG orientations
		pix_per_cell = 8 # HOG pixels per cell
		cell_per_block = 4 # HOG cells per block
		hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
		spatial_size = (32, 32) # Spatial binning dimensions
		hist_bins = 128    # Number of histogram bins
		spatial_feat = True # Spatial features on or off
		hist_feat = True # Histogram features on or off
		hog_feat = True # HOG features on or off
		y_start_stop = [400, 656] # Min and max in y to search in slide_window()


		utils = Utils()
		car_features = utils.extract_features(car_images_files, color_space, spatial_size,
								hist_bins, orient, 
								pix_per_cell, cell_per_block, hog_channel,
								spatial_feat, hist_feat, hog_feat)
		not_car_features = utils.extract_features(not_car_images_files, color_space, spatial_size,
								hist_bins, orient, 
								pix_per_cell, cell_per_block, hog_channel,
								spatial_feat, hist_feat, hog_feat)

		print (len(car_features))
		print (len(not_car_features))
		X = np.vstack((car_features, not_car_features)).astype(np.float64)
		print (X.shape)
		# Fit a per-column scaler
		X_scaler = StandardScaler().fit(X)
		# Apply the scaler to X
		scaled_X = X_scaler.transform(X)

		# Define the labels vector
		y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

		# Split up data into randomized training and test sets
		rand_state = np.random.randint(0, 100)
		X_train, X_test, y_train, y_test = train_test_split(
			scaled_X, y, test_size=0.2, random_state=rand_state)
		print("Feature vector length: ", len(X_train[0]))
		print("Count of Train data: ", len(X_train))

		## Use Grid Search CV to find best classifier parameters.
		#pipe, param_grid = self.pcaAndSVCPipe()
		#clf = GridSearchCV(pipe, param_grid = param_grid)

		# Use a linear SVC 
		clf = LinearSVC()
		# Check the training time for the SVC
		t=time.time()
		#svc.fit(X_train, y_train)
		clf.fit(X_train, y_train)
		#print (clf.best_estimator_)
		t2 = time.time()
		print (round(t2 - t), " seconds to train SVC")
		#accuracy = round(svc.score(X_test, y_test), 4)
		y_predict = clf.predict(X_test)
		self.evaluateModel(y_test, y_predict)
		#print ("Accuracy of the model is {:0.2f}".format(accuracy))
		return clf, X_scaler

	# Reduce dimensionality using , percentile and estimate with SVM 
	def pcaAndSVCPipe(self):
		# Define pipeline with PCA as feature selection and classifier
		
		pipe = Pipeline([
			('reduce_dim', SelectPercentile(f_classif)),
			('classify', LinearSVC())
		])
		
		
		param_grid = [
			{        
				'reduce_dim__percentile': [80, 90, 100],
				#'classify__kernel': 'linear', 'rbf'
				'classify__penalty':['l1', 'l2'], 
				'classify__loss': ['hinge', 'squared_hinge']
			},
		]
		return pipe, param_grid



	## Calculate accuracy, precision and recall
	def evaluateModel(self, labels_test, labels_predict):
		print ("Accuracy is {:0.2f}".format(float(accuracy_score(labels_test, labels_predict))))
		precision = float(precision_score(labels_test, labels_predict))
		recall = float(recall_score(labels_test, labels_predict))
		#score = float(f1_score(labels_test, labels_predict))    
		print ("Precision score {:0.2f} , recall score {:0.2f}".format(precision, recall))
