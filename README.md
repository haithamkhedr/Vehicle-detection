# Vehicle detection
The goal of this project is to detect and track vehicles in the scene as this is essential for self-driving cars

## Project steps
This is an explanation of the steps done in the project given a labeled dataset of images including cars and non cars. The dataset is a collection of Images from the Kitti and GTI datasets

#### 1-Feature extraction
Extraction of Hog features from the images together with resized spatial information(pixel values) and color histogram of the images in YCrCB color space

#### 2-Classification
Train an SVM linear classifier on the extracted features after feature normalization

#### 3-Sliding window
Implement a sliding window that passes through the image, extract features from the window and use the classifier to classify if there is a car inside the window or not.

