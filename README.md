## Validating COVID Rapid Test Results

### Objective
* COVID rapid tests will continue to play a key role in the post-pandemic world 
* Our project aims to develop a digital self-validation program for COVID rapid tests
* This involves recognising handwritten serial numbers and test results

![COVID Rapid Test Results](https://github.com/smkerr/smkerr.github.io/blob/main/assets/img/validating-covid-test-results.png?raw=true)

### Minimum Viable Product
1. Recognise serial number from image
1. Recognise COVID test result from image 

### Data 
We use data from the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/), a collection of more than 60,000 examples, to train our Machine Learning algorithm. 

### Dependencies
* Python ≥3.5
* sklearn ≥0.20
* cv2
* imutils

### Contributors 
* Steven Kerr
* Kai Foerster
* Dominik Cramer

### Sources
* [imutils package by PyImageSearch](https://github.com/PyImageSearch/imutils/blob/master/demos/sorting_contours.py)
* [Changing Colorspaces by OpenCV](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
* [Finding red color in image using Python & OpenCV from Stack Overflow](https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv)
* [How can I find contours inside ROI using opencv and Python? from Stack Overflow](https://stackoverflow.com/questions/42004652/how-can-i-find-contours-inside-roi-using-opencv-and-python)
* [What does bitwise_and operator exactly do in openCV? from Stack Overflow](https://stackoverflow.com/questions/44333605/what-does-bitwise-and-operator-exactly-do-in-opencv)
* [Resize an image without distortion OpenCV from Stack Overflow](https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv)
* [Wrong contours and wrong output of handwritten digit recognition AI model from Stack Overflow](https://stackoverflow.com/questions/62247234/wrong-contours-and-wrong-output-of-handwritten-digit-recognition-ai-model)
* [Building an Ensemble Learning Model Using Scikit-learn by Eijaz Allibhai](https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a)

### License
The material in this repository is made available under the [MIT license](https://github.com/smkerr/DSA--Final-Project/blob/main/LICENSE).
