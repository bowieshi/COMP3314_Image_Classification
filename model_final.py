from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

import gc
import argparse
import random
import math
import csv
import pickle
import cv2
import pywt
import mahotas
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import skimage
from skimage import color
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from skimage.feature import local_binary_pattern
from scipy.signal import convolve2d

# def TopColors_FeatureExtractor(image, num_colors):
#     # Reshape the image to a flattened array of pixels
#     pixels = image.reshape(-1, 3)
#
#     # Convert the pixel values to floating point
#     pixels = pixels.astype(np.float32)
#
#     # Perform k-means clustering to extract dominant colors
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
#     _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#
#     # Convert the color centers to the 8-bit unsigned integer format
#     centers = np.uint8(centers)
#
#     # Count the occurrences of each color
#     counts = np.bincount(labels.flatten())
#
#     # Sort the colors based on their frequency in descending order
#     sorted_centers = centers[np.argsort(-counts)]
#     top_colors = sorted_centers[:num_colors]
#     normalized_top_colors = top_colors / 255.0
#     return np.array(normalized_top_colors).reshape(-1)


class FeatureExtractor:
    def __init__(self):
        pass

    def predict(self, img):
        pass

    def featureExtract(self, imgs):
        feature = []
        for i in tqdm(range(0, imgs.shape[0])):
            dense_feat = self.predict(imgs[i])
            feature.append(dense_feat)

        scaler = StandardScaler()
        scaler.fit(feature)
        normalized_feature = scaler.transform(feature)

        return np.array(normalized_feature)


class Color(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        # img = img.reshape(img.shape[0] * img.shape[1], -1)
        # return np.mean(img, axis=0)

        pixels = np.array(img)

        # Quantize colors to, say, 64 colors
        # Reduce each color channel into 4 bins (4*4*4 = 64 colors)
        pixels_quantized = (pixels // 64) * 64

        # Create histogram
        hist, _ = np.histogramdd(pixels_quantized.reshape(-1, 3), bins=(4, 4, 4), range=((0, 256), (0, 256), (0, 256)))

        # Flatten histogram to use as a feature vector
        feature_vector = hist.flatten()

        # Normalize the histogram
        feature_vector /= feature_vector.sum()

        return feature_vector


class Raw(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        return img.flatten()


class SmoothRaw(FeatureExtractor):

    def __init__(self, size=16):
        super().__init__()
        self.size = size

    def predict(self, img):
        b, g, r = cv2.split(img)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        resized_grey_image = cv2.resize(gray_image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        resized_b = cv2.resize(b, (self.size, self.size), interpolation=cv2.INTER_AREA)
        resized_g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA)
        resized_r = cv2.resize(r, (self.size, self.size), interpolation=cv2.INTER_AREA)

        return np.concatenate((resized_grey_image, resized_b, resized_g, resized_r)).flatten()


class cv2HOG(FeatureExtractor):
    def __init__(self, win_size=(32, 32), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), num_bins=25):
        super().__init__()
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.num_bins = num_bins
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

    def predict(self, img):
        b_img, g_img, r_img = cv2.split(img)
        grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features_b = self.hog.compute(b_img)
        features_g = self.hog.compute(g_img)
        features_r = self.hog.compute(r_img)
        features_grey = self.hog.compute(grey_image)

        return np.concatenate((features_grey, features_b, features_g, features_r), axis=0)


class sklearnHOG(FeatureExtractor):
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=2):
        super().__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.channel_axis = channel_axis

    def predict(self, img):
        # b_img, g_img, r_img = cv2.split(img)
        # grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature = hog(img, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block, channel_axis=self.channel_axis)
        # features_b = self.hog.compute(b_img)
        # features_g = self.hog.compute(g_img)
        # features_r = self.hog.compute(r_img)
        # features_grey = self.hog.compute(grey_image)

        return feature


class SIFT(FeatureExtractor):

    def __init__(self):
        super().__init__()

    def predict(self, img):
        gray_im = color.rgb2gray(img)
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute((gray_im * 255).astype("uint8"), None)

        if descriptors is None:
            return np.zeros((128,))

        return np.mean(descriptors, axis=0)


class HuMoments(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(img_gray)
        humoments = cv2.HuMoments(moments)
        humoments = -(np.log(np.abs(humoments))) / np.log(10)
        return humoments.flatten()


class EOH(FeatureExtractor):
    def __init__(self, num_blocks=8):
        super().__init__()
        self.num_blocks = num_blocks

    def get_sobel_filters(self):
        # Define Sobel filters for horizontal, vertical, 45-degree, and 135-degree edges
        return [
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),  # Horizontal
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),  # Vertical
            np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32),  # 45-degree
            np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)  # 135-degree
        ]

    def apply_filters(self, image, filters):
        # Apply each filter to the image
        edge_maps = [convolve2d(image, filter, mode='same', boundary='wrap') for filter in filters]
        return edge_maps

    def calculate_histograms(self, image, edge_maps):
        # Assume the image is divided into num_blocks x num_blocks
        block_size = image.shape[0] // self.num_blocks
        histograms = np.zeros((self.num_blocks, self.num_blocks, len(edge_maps)), dtype=np.float32)

        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                block = [em[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] for em in
                         edge_maps]
                histograms[i, j] = [np.sum(np.abs(b)) for b in block]

        return histograms

    def predict(self, img):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get the Sobel filters
        filters = self.get_sobel_filters()

        # Apply the filters to the image
        edge_maps = self.apply_filters(gray, filters)

        # Calculate the histograms
        histograms = self.calculate_histograms(gray, edge_maps)

        # normalize the histograms
        histograms /= histograms.sum(axis=(0, 1), keepdims=True)

        # Flatten the histograms
        return histograms.flatten()


class LBP(FeatureExtractor):
    def __init__(self, num_points=8, radius=1):
        super().__init__()
        self.num_points = num_points
        self.radius = radius

    def predict(self, img, P=8, R=1):
        # Convert the image to grayscale as LBP works on grayscale images
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute the LBP features
        lbp_image = local_binary_pattern(gray_image, 8, 1, method='uniform')

        # Compute the histogram of LBP features
        histogram, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 60), range=(0, 59))

        # Normalize the histogram
        histogram = histogram.astype(np.float32)
        histogram /= (histogram.sum() + 1e-5)

        return histogram


class LDA(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, image):
        # Step 1: Flatten the image
        flattened_image = image.flatten()

        # Step 3: Perform LDA
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit_transform(flattened_image.reshape(1, -1), labels)

        # Extract the LDA feature in one dimension
        lda_feature = lda.transform(flattened_image.reshape(1, -1))

        return lda_feature

# class Gabor(FeatureExtractor):
#     def __init__(self, kernel_size=15, theta=np.pi/4, sigma=2.0, lambda_=5.0, gamma=0.5):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.theta = theta
#         self.sigma = sigma
#         self.lambda_ = lambda_
#         self.gamma = gamma
#     def predict(self, kernel_size, theta, sigma, lambda_, gamma):
#         # Generate a complex Gabor filter
#         kernel = cv2.getGaborKernel((self.kernel_size, self.kernel_size), self.sigma, self.theta, self.lambda_, self.gamma, 0, ktype=cv2.CV_32F)
#
#         # Normalize the filter
#         kernel /= np.sqrt((kernel * kernel).sum())
#
#         return kernel

class Haar(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self,img):
        # Load the Haar cascade classifier
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)

        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect Haar-like features
        haar_features = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Convert the Haar-like features to a vector
        feature_vector = np.array([])
        for (x, y, w, h) in haar_features:
            feature_vector = np.concatenate((feature_vector, [x, y, w, h]))

        return feature_vector

class cooccurrence(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        co_matrix = skimage.feature.graycomatrix(img, [5], [0], levels=256, symmetric=True, normed=True)

        # Calculate texture features from the co-occurrence matrix
        contrast = skimage.feature.graycoprops(co_matrix, 'contrast')
        correlation = skimage.feature.graycoprops(co_matrix, 'correlation')
        energy = skimage.feature.graycoprops(co_matrix, 'energy')
        homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')

        return np.array([contrast[0, 0], correlation[0, 0], energy[0, 0], homogeneity[0, 0]]).flatten()


class LPQ(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute local phase
        kernel_size = 3
        lpq = cv2.phase(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel_size),
                        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel_size), angleInDegrees=True)

        # Quantize the local phase
        threshold = np.median(lpq)
        lpq_binary = (lpq > threshold).astype(np.uint8)

        # Compute LPQ feature vector
        feature_vector = np.concatenate((np.histogram(lpq_binary, bins=256, range=(0, 256))[0],
                                         np.histogram(lpq, bins=256, range=(0, 256))[0]))

        return feature_vector


class ColorHistogram(FeatureExtractor):
    def __init__(self, num_bins=60):
        super().__init__()
        self.num_bins = num_bins

    def predict(self, image):
        # Convert the image from BGR to RGB color space
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate the histogram for each color channel
        hist_red = cv2.calcHist([rgb_image], [0], None, [self.num_bins], [0, 256])
        hist_green = cv2.calcHist([rgb_image], [1], None, [self.num_bins], [0, 256])
        hist_blue = cv2.calcHist([rgb_image], [2], None, [self.num_bins], [0, 256])

        # Normalize the histograms
        cv2.normalize(hist_red, hist_red, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_green, hist_green, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_blue, hist_blue, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Concatenate the histograms into a single feature vector
        color_histogram = np.concatenate([hist_red, hist_green, hist_blue]).flatten()

        return color_histogram


class Tamura(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the coarseness, contrast, and directionality
        coarseness = self.calculate_coarseness(img)
        contrast = self.calculate_contrast(img)
        directionality = self.calculate_directionality(img)

        return np.array([coarseness, contrast, directionality]).flatten()

    def calculate_coarseness(self, img):
        # Calculate the gradient of the image
        gradient = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)

        # Calculate the mean of the gradient
        mean = np.abs(gradient).mean()

        # Calculate the coarseness
        coarseness = mean / 4.0

        return coarseness

    def calculate_contrast(self, img):
        # Calculate the contrast
        contrast = img.std()

        return contrast

    def calculate_directionality(self, img):
        # Calculate the gradient of the image
        gradient = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)

        # Calculate the histogram of the gradient
        histogram, _ = np.histogram(gradient.ravel(), bins=256, range=(-128, 128))

        # Normalize the histogram
        histogram = histogram.astype(np.float32)
        histogram /= (histogram.sum() + 1e-5)

        # Calculate the entropy of the histogram
        entropy = -np.sum(histogram * np.log2(histogram + 1e-5))

        # Calculate the directionality
        directionality = 1 - 1 / (1 + entropy)

        return directionality


class Daubechies_Wavelets(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform the Daubechies wavelet transform
        coeffs = pywt.dwt2(img, 'db1')

        # Calculate the mean and standard deviation of the coefficients
        mean = [np.mean(c) for c in coeffs]
        std = [np.std(c) for c in coeffs]

        return np.concatenate((mean, std)).flatten()


class ZernikeMoments(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the Zernike moments
        moments = mahotas.features.zernike_moments(img, radius=21, degree=8)

        return moments.flatten()


class Haralick(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the Haralick texture features
        features = mahotas.features.haralick(img).mean(axis=0)

        return features.flatten()


class GIST(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the GIST features
        features = mahotas.features.haralick(img).mean(axis=0)

        return features.flatten()


class DenseSIFT(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        gray_im = color.rgb2gray(img)
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute((gray_im * 255).astype("uint8"), None)

        if descriptors is None:
            return np.zeros((128,))

        return np.mean(descriptors, axis=0)


class DCT(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def predict(self, img):
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform the 2D DCT
        dct = cv2.dct(np.float32(img))

        return dct.flatten()


def random_rotate(img, degrees_range=(-10, 10)):
    angle = random.uniform(degrees_range[0], degrees_range[1])
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_image


def extract_catgory_data(train_imgs, train_labels, val_imgs, val_labels, category):
    category_train_imgs = []
    category_train_labels = []
    category_val_imgs = []
    category_val_labels = []
    for i in range(len(train_imgs)):
        if train_labels[i] in category:
            category_train_imgs.append(train_imgs[i])
            category_train_labels.append(train_labels[i])
    for i in range(len(val_imgs)):
        if val_labels[i] in category:
            category_val_imgs.append(val_imgs[i])
            category_val_labels.append(val_labels[i])
    return category_train_imgs, category_train_labels, category_val_imgs, category_val_labels


image_names = []
train_labels = []
test_image_names = []
test_labels = []

train_csv_file = './train.csv'
test_csv_file = './test.csv'

with open(train_csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        image_names.append(row[0])
        train_labels.append(int(row[1]))

train_imgs = []
for i, image_name in enumerate(image_names):
    images_path = f'./train_ims/{image_name}'
    img = cv2.imread(images_path)
    train_imgs.append(img)

with open(test_csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        test_image_names.append(row[0])
        test_labels.append(int(row[1]))

test_imgs = []
for i, image_name in enumerate(test_image_names):
    images_path = f'./test_ims/{image_name}'
    img = cv2.imread(images_path)
    test_imgs.append(img)

print("finished loading")

print("begin data augmentation")
size = len(train_imgs)
for i in range(size):
    img = train_imgs[i]
    # flip
    img = cv2.flip(img, 1)
    train_imgs.append(img)
    train_labels.append(train_labels[i])

print("data augmentation done")

train_size = len(train_imgs)
test_size = len(test_labels)

print("train size:", train_size)
for i in range(10):
    print("Number of images with label", i, ":", train_labels.count(i))
print("val size:", test_size)

imgs = np.array(train_imgs + test_imgs)
labels = np.array(train_labels + test_labels)

# Gabor_extractor = Gabor()
# ZernikeMoments_extractor = ZernikeMoments()
# Haralick_extractor = Haralick()
Daubechies_Wavelets_extractor = Daubechies_Wavelets()
Tamura_extractor = Tamura()
ColorHistogram_extractor = ColorHistogram()
LPQ_extractor = LPQ()
cooccurrence_extractor = cooccurrence()
cvHOG_extractor = cv2HOG()
Color_extractor = Color()
Raw_extractor = Raw()
EOH_extractor = EOH()
HuMoments_extractor = HuMoments()
SIFT_extractor = SIFT()
LBP_extractor = LBP()

# Gabor_feature = features_file['Gabor_feature']
# Tamura_feature = features_file['Tamura_feature']
# cooccurrence_feature = features_file['cooccurrence_feature']
# Daubechies_Wavelets_feature = features_file['Daubechies_Wavelets_feature']
# ZernikeMoments_feature = features_file['ZernikeMoments_feature']
# Haralick_feature = features_file['Haralick_feature']
# GIST_feature = features_file['GIST_feature']

# Gabor_feature = Gabor_extractor.featureExtract(imgs)
# ZernikeMoments_feature = ZernikeMoments_extractor.featureExtract(imgs)
# Haralick_feature = Haralick_extractor.featureExtract(imgs)
Daubechies_Wavelets_feature = Daubechies_Wavelets_extractor.featureExtract(imgs)
Tamura_feature = Tamura_extractor.featureExtract(imgs)
ColorHistogram_feature = ColorHistogram_extractor.featureExtract(imgs)
LPQ_feature = LPQ_extractor.featureExtract(imgs)
cooccurrence_feature = cooccurrence_extractor.featureExtract(imgs)
cvHOG_feature = cvHOG_extractor.featureExtract(imgs)
Color_feature = Color_extractor.featureExtract(imgs)
raw_feature = Raw_extractor.featureExtract(imgs)
EOH_feature = EOH_extractor.featureExtract(imgs)
HuMoments_feature = HuMoments_extractor.featureExtract(imgs)
SIFT_feature = SIFT_extractor.featureExtract(imgs)
LBP_feature = LBP_extractor.featureExtract(imgs)

imgs_features = np.concatenate(
    (cvHOG_feature, raw_feature, Color_feature, EOH_feature, HuMoments_feature, SIFT_feature, LBP_feature,
     cooccurrence_feature, LPQ_feature, ColorHistogram_feature, Tamura_feature, Daubechies_Wavelets_feature,),
    axis=1)
del cvHOG_feature
del raw_feature
del Color_feature
del EOH_feature
del HuMoments_feature
del SIFT_feature
del LBP_feature
gc.collect()

print(imgs_features.shape)

print("begin pca")
pca = PCA(0.87, copy=False)
pca.fit(imgs_features)
imgs_features = pca.fit_transform(imgs_features)
print("pca done")

train_features = imgs_features[:100000]
train_labels = labels[:100000]

print("begin training")
model = SVC(kernel='rbf', C=5, random_state=42)
model.fit(train_features, train_labels)
print("training done")

print("begin testing")
test_features = np.array(imgs_features[100000:])
test_predictions = model.predict(test_features)
print("testing done")

print("begin saving")
# Create a dataframe with the image names and predictions
result_data = pd.DataFrame({'im_name': test_image_names, 'label': test_predictions})

# Save the result to a CSV file
output_file_path = "predict.csv"
result_data.to_csv(output_file_path, index=False)
print("saving done")