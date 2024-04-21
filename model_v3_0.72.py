from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

import argparse
import random
import math
import csv
import pickle
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import color
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans


# Histogram of Oriented Gradients (HOG)
epsilon = 1e-5
n_clusters = 30
number_of_bins = 9
step_size = 180 / number_of_bins

# Extract the three most significant colors as features
# return (9, ) feature embedding
def TopColors_FeatureExtractor(image, num_colors):
    # Reshape the image to a flattened array of pixels
    pixels = image.reshape(-1, 3)

    # Convert the pixel values to floating point
    pixels = pixels.astype(np.float32)

    # Perform k-means clustering to extract dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the color centers to the 8-bit unsigned integer format
    centers = np.uint8(centers)

    # Count the occurrences of each color
    counts = np.bincount(labels.flatten())

    # Sort the colors based on their frequency in descending order
    sorted_centers = centers[np.argsort(-counts)]
    top_colors = sorted_centers[:num_colors]
    normalized_top_colors = top_colors / 255.0
    return np.array(normalized_top_colors).reshape(-1)


def smooth_raw(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (16, 16), interpolation=cv2.INTER_AREA)
    return resized_image.flatten() / 255.0

# Histogram of Oriented Gradients (HOG)
def HOG_FeatureExtractor(image):
    b, g, r = cv2.split(image)
    # Define the HOG descriptor with adjusted parameters
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    win_size = (32, 32)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    num_bins = 9
    # Extract features using HOG
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

    # Compute the HOG features for each color channel
    features_b = hog.compute(b)
    features_g = hog.compute(g)
    features_r = hog.compute(r)
    features_grey = hog.compute(grey_image)
    # print(features_b.shape, features_g.shape, features_r.shape)
    return np.concatenate((features_grey, features_b, features_g, features_r), axis=0)
    # return features_grey

def HOG_FeatureExtractor2(img, kmean=None):
    gray = color.rgb2gray(img) / 255
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), feature_vector=True)
    if kmean == None:
        return features
    else:
        cell_cat = kmean.predict(features.reshape(-1, 9))
        histogram, _ = np.histogram(cell_cat, bins=n_clusters)
        return histogram / len(cell_cat)

def HOG_KNN():
    cnt = 0
    train_features = []
    for image_name in image_names:
        images_path = f'./train_ims/{image_name}'
        img = cv2.imread(images_path) # (32, 32, 3)
        orient_histogram = HOG_FeatureExtractor(img)
        train_features.append(orient_histogram)
        cnt += 1
        if cnt % 1000 == 0:
            print("Processed", cnt, "images")
    train_features = np.array(train_features).reshape(-1, 9)
    print(train_features.shape)
    print("begin kmeans")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(train_features)
    print("kmeans done")
    return kmeans


def train_model():
    image_names = []
    labels = []

    csv_file = './train.csv'

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            image_names.append(row[0])
            labels.append(int(row[1]))

    # print("Image Names:", image_names)
    # print("Labels:", labels)
    for i in range(10):
        print("Number of images with label", i, ":", labels.count(i))

    labels = np.array(labels)

    data = list(zip(image_names, labels))
    random.shuffle(data)
    images_names_shuffled, labels_shuffled = zip(*data)

    # kmean = HOG_KNN()

    image_features = []

    for i, image_name in enumerate(image_names):

        images_path = f'./train_ims/{image_name}'
        image = cv2.imread(images_path)
        features = np.concatenate((smooth_raw(image), HOG_FeatureExtractor(image), TopColors_FeatureExtractor(image, 3)), axis=0)
        image_features.append(features)

        if ((i+1) % 1000 == 0):
            # print(features.shape)
            print("Processed", i, "images")

    image_features = np.array(image_features)
    print("begin split")
    # print(image_features.shape)
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_features, labels, test_size=0.1, random_state=42
    )
    print("split done")
    print("begin training")
    # kernel SVC
    model = SVC(kernel='rbf', max_iter=3000, gamma=0.09, C=10)
    model.fit(train_images, train_labels)
    print("training done")
    # # LinearSVC
    # model = LinearSVC(multi_class='ovr', max_iter=10000)
    # model.fit(train_images, train_labels)

    # Evaluate the model on the test dataset
    predictions = model.predict(test_images)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy}")

    # Save the SVM model parameters to a file
    filename = 'svm_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def test_model():

    image_names = []
    labels = []

    csv_file = './test.csv'

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            image_names.append(row[0])
            labels.append(int(row[1]))
    print("Number of test images:", len(image_names))
    model_file_path = "svm_model.pkl"
    with open(model_file_path, 'rb') as file:
        svc_model = pickle.load(file)

    image_features = []
    for i, image_name in enumerate(image_names):

        images_path = f'./test_ims/{image_name}'
        image = cv2.imread(images_path)
        features = np.concatenate((smooth_raw(image), HOG_FeatureExtractor(image), TopColors_FeatureExtractor(image, 3)), axis=0)
        image_features.append(features)

        if ((i+1) % 1000 == 0):
            # print(features.shape)
            print("Processed", i, "images")

    image_features = np.array(image_features)
    predictions = svc_model.predict(image_features)

    # Create a dataframe with the image names and predictions
    result_data = pd.DataFrame({'im_name': image_names, 'label': predictions})

    # Save the result to a CSV file
    output_file_path = "predict.csv"
    result_data.to_csv(output_file_path, index=False)

# Create an argument parser
parser = argparse.ArgumentParser(description='Train or test the model.')
parser.add_argument('mode', choices=['train', 'test'], help='Specify whether to train or test the model.')

# Parse the command-line arguments
args = parser.parse_args()

if args.mode == 'train':
    train_model()
elif args.mode == 'test':
    test_model()
else:
    print('Invalid mode specified. Please choose either "train" or "test".')