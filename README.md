# Vehicle detection
The goal of this project is to detect and track vehicles in the scene as this is essential for self-driving cars

## Project steps
This is an explanation of the steps done in the project given a labeled dataset of images including cars and non cars. The dataset is a collection of Images from the Kitti and GTI datasets

#### 1-Feature extraction
Extraction of Hog features from the images together with resized spatial information(pixel values) and color histogram of the images in YCrCb color space

#### 2-Classification
Train an SVM linear classifier on the extracted features after feature normalization

#### 3-Sliding window
Implement a sliding window that passes through the image, extract features from the window and use the classifier to classify if there is a car inside the window or not.


## Histogram of Oriented Gradients (HOG)
The code for Hog features extraction is found in `features.py` in the function `get_hog_features()` line #59.
I used `skimage.hog()` to calulcate the HOG features for the image after reading it.
I tried extracting the Hog features from different color spaces like `HSV` and `YCrCb` and found that using YCrCb color space minimized the false positives detection.The following images shows an example of the extracted Hog features using YCrCb color space.
![ScreenShot] (./output_images/Hog.jpg)

Regarding the hog parameters, I manually tuned the 3 hog parameters which are `orientations`,`pix_per_cell`,`cell_per_block`. I tried orientations value from 9 to 12 , pix_per_cell from 7 to 10 but they did not affect performance so I chose to use 9 `orientations`, 8 `pix_per_cell` and 2 `cell_per_block` . The most important thing that affected the classification accuracy is the number of channels to extract hog features from. I tried many combinations but found that using all channels increased classification accuracy by 2 %.

## Training a classifier
This part is found in the notebook `main/Vehicle Detection.ipynb` from cell 3 to cell 9. Starting from cell 3, features are extracted from the dataset and saved to a pickle file, after that the data is randomly shuffled to prevent overfitting and to increase training speed, then the data is standardized(each feature has zero mean and unit standard deviation) and split into training and test sets, finally the training set is fed to the classifier to train on. I used a simple linear SVM as my classifier and it achieved an excellent accuracy (99%) on my test set.

## Sliding window implementation
The sliding window is implemented in `main/Vehicle Detection.ipynb` cell #11 in the function `find_cars()`
The idea here is to pass a sliding window across the image with different scales and search for cars in each window.
There are 2 methods to implement the sliding window method, the first is to extract the hog -and other- features for each window then classify it. The other method is to compute the hog features for the whole image only once and then extract subarrays(features) for each window.
I used the 2nd method because it is faster. The implementation of the sliding window needs a lot of tuning to decide how much windows overlap? and the scales of the sliding windows. First of all I tried overlap of 30-50-75 % and only the 75 % overlapping window gave a good result. I used 3 different sliding windows with sizes 108,96,87. The decision of such scales required too much manual tuning to achieve a good result. The last trick used to decrease the number of search windows is to search in the bottom half of the image as there won't be any cars in the sky. The image below show all the 108 X 108 windows being searched.
![ScreenShot] (output_images/windows.jpg)

## Minimizing false positives and examples
There are 3 methods that worked with me to minimize false positives. The first is using YCrCb color space features instead of HSV, this enhanced my pipeline alot. The 2nd method is constructing a heatmap using the findings of the windows of different scales and then thresholding this heatmap, this is based on the consideration that a false positive may be found once or twice only in a single frame and hence can be threhsolded out. The 3rd method is simply ruling out bounding boxes(detections after thresholding heatmap) that are too small. Here are some examples that shows the output of the pipeline in different images.
![ScreenShot] (output_images/example1.jpg)
![ScreenShot] (output_images/example2.jpg)
![ScreenShot] (output_images/example3.jpg)

## Video
Here is a [link](https://youtu.be/R0ns2lJjJiY) to the video output

#### False positives filtering
I used the same technique of heatmaps as described before to minimize the false positives but integrated it along several(10) frames.
This part is implemented in `main/Vehicle Detection.ipynb` in the cell before last. A buffer `cumulative_heat` holds the heatmaps of the latest 10 frames and for each new frame the heatmap is computed as a weighted average between the current heat map and the average heatmap of the 10 previous frames as shown below where `alpha` was set to 0.7 and `beta` was set to 0.3
```
heatmap  = beta * frame_heat + alpha * np.mean(cumulative_heat,axis = 0)
heatmap = heatmap_threshold(heatmap , threshold = 2.5)
```

## Discussion
Regarding the false positives, I think one way for minimizing them is to add more training data for the `not car` class.
I also think that the vision approach alone won't be very robust against different conditions. To make it more robust we can add a kalman filter to track the movement of the cars.
Overall this was an amazing project
