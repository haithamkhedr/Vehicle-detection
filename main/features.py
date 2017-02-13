import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import glob
import pickle

def exploreData():
    file_names = glob.glob("../Data/*/*/*.png")
    nImgs = len(file_names)
    img = cv2.imread(file_names[0])
    shape = img.shape
    print("Number of Training samples: ",nImgs,"\nImage shape: ",shape)


####Resize image####
def compress_spatially(img,size = (32,32) , feature_vec = True):
    img = cv2.resize(img,size)
    if feature_vec is True:
        img = img.ravel()
    return img



####Color space conversion####
##Assume input is BGR
def transform_color(img,space = 'BGR'):

    if space is 'BGR':
        transformed = np.copy(img)
    elif space is 'RGB':
        transformed = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    elif space is 'HSV':
        transformed = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    elif space is 'HLS':
        transformed = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    elif space is 'YUV':
        transformed = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    elif space is 'GRAY':
        transformed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return transformed

## Returns color histogram
def color_histogram(img , nbins = 32 , bins_range = (0,256)):
    if len(img.shape) < 3:
        features = np.histogram(img, bins = nbins , range = bins_range)

    else:
        rhist1 = np.histogram(img[:,:,0] , bins = nbins , range = bins_range)
        rhist2 = np.histogram(img[:,:,1] , bins = nbins , range = bins_range)
        rhist3 = np.histogram(img[:,:,2] , bins = nbins , range = bins_range)
        features = np.concatenate((rhist1[0],rhist2[0],rhist3[0]))

    return features

###calculates the Hog features for a single channel
def get_hog_features(img,orient = 9,pix_per_cell = 8 , cell_per_block = 2 ,vis = False , feature_vec = True):

    if vis is True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features,hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


##calculates HOG features for single/multi channels
def hog_features(img,orient = 9,pix_per_cell = 8 , cell_per_block = 2 , feature_vec = True , channel = 'ALL'):
    vis = False
    features = []
    if channel is 'ALL':
        for ch in range(3):
            hog_features = hog(img[:,:,ch], orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            features.append(hog_features)

    else:
        features = hog(img[:,:,channel], orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)

    return features

###### Extract features from Raw BGR image######
def extract_features(directory, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel = 'ALL',
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    labels = []
    for imgPath in directory:
        if 'non-vehicles' in  imgPath:
            labels.append(0)
        else:
            labels.append(1)
        img_features = []
        img = cv2.imread(imgPath)
        img = transform_color(img , space = color_space)
        if spatial_feat is True:
            spatial_features = compress_spatially(img,size = (32,32))
            img_features.append(spatial_features)
            #print("spatial features:",len(spatial_features))
        if hist_feat is True:
            color_features = color_histogram(img , nbins = hist_bins)
            img_features.append(color_features)
            #print("color features:",len(color_features))
        if hog_feat is True:
            HOG_features = hog_features(img,orient ,pix_per_cell , cell_per_block , feature_vec = True , channel = hog_channel)
            HOG_features = np.ravel(HOG_features)
            img_features.append(HOG_features)
            #print("Hog features:",len(HOG_features))

        img_features = np.concatenate(img_features)
        features.append(img_features)
    features = np.array(features)
    labels = np.array(labels)
    data = {'data':features , 'labels':labels}
    pickle.dump(data , open('Data.p','wb'))
    print('Features are extracted and saved to Data.p')
    print('Data set size : ', features.shape)
    return data

