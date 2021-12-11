# -*- coding: utf-8 -*-
"""Recognising COVID Rapid Test Results with Python

Key Assumptions:
* Our code works for one type of COVID self test
* The COVID test results will display in a pink or red hue

Table of Contents
1. Repository Structure
2. Import Packages
3. Train Algorithm
4. Create Ensemble Model
5. Load Image
6. Recognise COVID Test Results
7. Recognise Handwritten Serial Number


1. Repository Structure

DSA--Final-Project/
│
├── project/ 
├── test-images/ 
├── .gitignore 
├── README
├── LICENSE


2. Import Dependencies

Python ≥3.5 and Scikit-Learn ≥0.20 are required for this project. Additional packages are imported and a seed is set to make this
notebook's output stable across runs.
"""

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# import necessary dependencies
import os
import numpy as np
import cv2
import imutils
from imutils import contours

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# set seed to make this notebook's output stable across runs
np.random.seed(42)

"""3. Train Algorithm

The MNIST dataset containing images of handwritten digits is imported and then randomly split into a training set and a testing set. 
Then, the Multi-layer Perception classifier, the K-Nearest Neighbor classifier, and the Random Forest classifier are imported and trained
individually before predictions are made and an accuracy score for each of them is calculated and printed.
"""

# import dataset 
from sklearn.datasets import fetch_openml

# import 28x28 images of handwritten digits from MNIST database
mnist = fetch_openml('mnist_784', version = 1, as_frame = False)
mnist.target = mnist.target.astype(np.uint8) # as uint8 data type

# randomly split dataset into training set and testing set 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size = 10000, random_state = 42)

# import classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# train Multi-layer Perception classifier
mlp_clr = MLPClassifier(random_state = 42)
mlp_clr.fit(x_train, y_train)

# train K Nearest Neighbor classifier
knn_clr = KNeighborsClassifier(n_neighbors = 3)
knn_clr.fit(x_train, y_train)

# train Random Forest classifier
rf_clr = RandomForestClassifier(random_state = 42, n_estimators = 100)
rf_clr.fit(x_train, y_train)

# make predictions with individual classifiers
y_pred_mlp = mlp_clr.predict(x_test)
y_pred_knn = knn_clr.predict(x_test)
y_pred_rf = rf_clr.predict(x_test)

# calculate accuracy scores for various classifiers
from sklearn.metrics import accuracy_score

print("ACCURACY SCORES")
print("---------------")
print("MLP classifier: ", accuracy_score(y_test, y_pred_mlp))
print("KNN classifier: ", accuracy_score(y_test, y_pred_knn))
print("RF classifier:  ", accuracy_score(y_test, y_pred_rf))

"""4. Create Ensemble Model

After an ensemble classifier is imported, the three classifiers trained above are combined in order to create a more accurate ensemble model. 
This is then used to make a prediction and once again calculate and print accuracy scores.
"""

# import ensemble classifier
from sklearn.ensemble import VotingClassifier

# combine classifiers to create a more accurate ensemble classifier
vot_clr = VotingClassifier(estimators = [('mlp', mlp_clr), ('knn', knn_clr),('rf', rf_clr)],
                           voting = 'soft') # voting classifier predicts based on the argmax of the sums of classifiers' predicted probabilities 
vot_clr.fit(x_train, y_train)

# make prediction with ensemble classifier
y_pred_vot = vot_clr.predict(x_test)

print("ACCURACY SCORES")
print("---------------")
print("MLP classifier: ", accuracy_score(y_test, y_pred_mlp))
print("KNN classifier: ", accuracy_score(y_test, y_pred_knn))
print("RF classifier:  ", accuracy_score(y_test, y_pred_rf))
print("*Voting classifier: ", accuracy_score(y_test, y_pred_vot))

"""5. Load Image

An image containing an antigen rapid test is loaded from the repository. It is then resized and displayed.
"""

# load image 
orig = cv2.imread("./test-images/test-image-1.png") # can also try test-image_2.png or test-image-3.png

# resize image to standardize 
new_h = 500 #
h, w = orig.shape[:2]
r =  new_h / float(h)
dim = (int(w * r), new_h)
img = resized = cv2.resize(orig, dim, interpolation = cv2.INTER_AREA)

# display image 
cv2.imshow('image', img)

"""6. Recognise COVID Test Results

First, the BGR image is converted to HSV colour space. The distinct pink/red hue present in the control- and test-lines of COVID 
antigen tests is used to create a mask. The mask highlights potential control- and test-lines present in the image (if any. The potential 
control- and test-lines are then converted to grayscale, blurred, and thresholded to further refine the computer's vision. The remaining 
contours are located, counted, and drawn before the test is classified as positive, negative, or inconclusive depending on the number of 
contours present.
"""

# convert the BGR image to HSV colour space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# display image 
cv2.imshow('hsv', hsv)

# set the lower and upper bounds for the red hue
lower_red = np.array([150,60,0])
upper_red = np.array([179,255,255])

# create a mask for red colour using inRange function
mask = cv2.inRange(hsv, lower_red, upper_red)

# display image 
cv2.imshow('mask', mask)

# perform bitwise and on the original image arrays using the mask
res = cv2.bitwise_and(img, img, mask = mask)

# display image
cv2.imshow('res', res)

# gray
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

# display image
cv2.imshow('gray', gray)

# blur 
blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Gaussian blurring with a 5×5 kernel to reduce high-frequency noise

# display image
cv2.imshow('blurred', blurred)

# thresh
ret, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

# display image 
cv2.imshow('thresh', thresh)

# find contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)

# draw contours
cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

# display image
cv2.imshow('image', img)

# drumroll please...
if len(cnts) == 2:
  print('The number of lines found in the COVID test is', len(cnts) )
  print('The COVID test is positive')
elif len(cnts) == 1:
  print('The number of lines found in the COVID test is', len(cnts) )
  print('The COVID test is negative')
else: 
  print('The number of lines found in the COVID test is', len(cnts) )
  print('The COVID test is inconclusive')

"""7. Recognise Handwritten Serial Numbers

A shape of black pixels is created in the same size as the original resized image. Since the serial number ('region of interest')
is located towards the bottom of the COVID test, a mask is created to highlight the region of interest. The original resized image is 
then grayed, blurred, and thresholded before the mask is applied. The remaining contours (ideally the digits) are sorted from left-to-right 
and a list is initialized to which the digit arrays are appended once the rectangle points for each digit contour have been extracted, padding 
around the digit array has been added, and the digit contour arrays have been resized. Afterwards, the list of digits is converted to np.array 
and reshaped. Finally, the ensemble classifier predicts the value of each digit contour in the list. 
"""

# reset image 
img = resized

# construct an image of black pixels of the same size.
black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8) #black in RGB

# display image 
cv2.imshow('black', black)

# form the mask and highlight the ROI:
black = cv2.rectangle(black,(10, 445),(135, 485),(255, 255, 255), -1) # the dimension of the ROI
gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY) # convert to gray
ret, b_mask = cv2.threshold(gray, 0, 255, 0) # convert to binary

# display image 
cv2.imshow('b_mask', b_mask)

# process original image 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray
cv2.imshow('gray', gray)

blurred = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT) # Gaussian blurring with a 5×5 kernel to reduce high-frequency noise
cv2.imshow('blurred', blurred)

ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV) # thresh
cv2.imshow('thresh', thresh)

# Mask the image above with your original image
masked = cv2.bitwise_and(thresh, thresh, mask = b_mask)

# display image
cv2.imshow('masked', masked)

# find contours
cnts = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #or try cv2.CHAIN_APPROX_SIMPLE
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right
(cnts, _) = contours.sort_contours(cnts, method = "left-to-right")

# initialize list of digit arrays
digits = []
for c in cnts:
   x,y,w,h = cv2.boundingRect(c) # extract bounding rectangle points for each digit contour
   cv2.rectangle(img, # image
                 (x, y), # start point
                 (x+w, y+h), # end point 
                 (0, 255, 0), # color (green)
                 2) # thickness
   digit = thresh[y:y+h, x:x+w] # threshold the digit
   padded_digit = np.pad(array = digit, 
                         pad_width = ((10,10), (10,10)), # add padding around digit array
                         mode = "constant", 
                         constant_values = 0)
   digit = cv2.resize(src = padded_digit, 
                      dsize = (28,28)) # desired size of the output image, given as tuple
   digits.append(digit) # creates a list of digit arrays

# display contours 
cv2.imshow('image', img)

# display first digit to make sure order is correct
cv2.imshow('first digit', digits[0])
digits[1].shape

digits = np.array(digits) # convert list of digits to np.array

digits = digits.reshape(digits.shape[0], # length remains equal to the number of digits
                        digits.shape[1]*digits.shape[2]) # the new size is the heighth x width of the digit images (28 x 28 = 784)

y_pred = vot_clr.predict(digits) # use ensemble classifier to identify serial number

print("The serial number for this COVID test is:", y_pred)
