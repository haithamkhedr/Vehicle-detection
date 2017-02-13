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


## Histogram of Oriented Gradients (HOG)
The code for Hog features extraction is found in `features.py` in the function `get_hog_features()` line #59.
I used `skimage.hog()` to calulcate the HOG features for and image after reading it.
I tried extracting the Hog features from different color spaces like `HSV` and `YCrCb` and found that using YCrCb color space achieved the best classification accuracy on the test set.The following images shows an example of the extracted Hog features.
![ScreenShot] (./output_images/Hog.jpg)
