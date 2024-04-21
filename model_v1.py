import numpy as np
import csv
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
print("read in image and labels")


# Histogram of Oriented Gradients (HOG)
def HOG_FeatureExtractor(image):
    b, g, r = cv2.split(image)
    # Define the HOG descriptor with adjusted parameters
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    win_size = (32, 32)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (4, 4)
    num_bins = 12
    # Extract features using HOG
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

    # Compute the HOG features for each color channel
    features_b = hog.compute(b)
    features_g = hog.compute(g)
    features_r = hog.compute(r)
    # features_grey = hog.compute(grey_image)
    # print(features_b.shape, features_g.shape, features_r.shape)
    return np.concatenate((features_b, features_g, features_r), axis=0)
    # return features_grey


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


import mahotas as mh


def HaralickTexture_FeatureExtractor(image, num_bins=16):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Haralick texture features
    textures = mh.features.haralick(gray)

    # Calculate the histogram of texture features
    histogram, _ = np.histogram(textures, bins=num_bins, range=(textures.min(), textures.max() + 1), density=True)

    return histogram


# Local Binary Patterns (LBP)
from skimage.feature import local_binary_pattern


def LBP_FeatureExtractor(image, num_points=8, radius=1, num_bins=256):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 32x32
    resized = cv2.resize(gray, (32, 32))

    # Calculate LBP
    lbp = local_binary_pattern(resized, num_points, radius, method='uniform')

    # Calculate histogram of LBP image
    histogram, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins - 1), density=True)

    return histogram


def Color_Entropy_FeatureExtractor(image, num_bins=256):
    # Convert the image to the HSV color space (hue saturation value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV image into individual channels
    h, s, v = cv2.split(hsv)

    # Calculate histograms of each color channel
    hist_h, _ = np.histogram(h, bins=num_bins, range=(0, 180), density=True)
    hist_s, _ = np.histogram(s, bins=num_bins, range=(0, 256), density=True)
    hist_v, _ = np.histogram(v, bins=num_bins, range=(0, 256), density=True)

    # Compute entropy of color histograms
    entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-7))
    entropy_s = -np.sum(hist_s * np.log2(hist_s + 1e-7))
    entropy_v = -np.sum(hist_v * np.log2(hist_v + 1e-7))

    # Combine the entropies into a single feature vector
    feature_vector = np.array([entropy_h, entropy_s, entropy_v])

    return feature_vector


def Haar_FeatureExtractor(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 32x32
    resized = cv2.resize(gray, (32, 32))

    # Calculate Haar-like features
    features = cv2.HaarEvaluator().compute(resized)

    return features


# Convert labels to a NumPy array
labels = np.array(labels)

# Initialize lists to store image features
image_features = []

print("Extracting feature...")
cnt = 0
# Process images and extract features
for image_name in image_names:
    # Load the image using OpenCV
    images_path = f'./train_ims/{image_name}'
    image = cv2.imread(images_path)

    HOG_feature = HOG_FeatureExtractor(image)
    # Haralick_feature = HaralickTexture_FeatureExtractor(image)
    # ColorEntropy_feature = Color_Entropy_FeatureExtractor(image)

    # Concatenate the features into a single feature vector
    features = np.concatenate((HOG_feature, ), axis=0)

    # features = np.concatenate((HOG_feature, ), axis=0)
    # features = np.concatenate((raw_feature, ), axis=0)
    # Store the features
    image_features.append(features)

    if (cnt % 1000 == 0):
        print("Processed", cnt, "images")
    cnt += 1

# Convert the list of features to a NumPy array
image_features = np.array(image_features)
train_images, test_images, train_labels, test_labels = train_test_split(
    image_features, labels, test_size=0.2, random_state=42
)


# kernel SVC
from sklearn.svm import SVC

model = [SVC(kernel='rbf', max_iter=3000) for i in range(10)]
for i in range(10):
    index_i = [x == i for x in train_labels]
    index_not_i = [x != i for x in train_labels]
    train_images_i = train_images[index_i]
    train_images_not_i = train_images[index_not_i]
    new_train_images = np.concatenate((train_images_i, train_images_not_i[:10000]), axis=0)
    new_train_labels = np.concatenate((np.ones(train_images_i.shape[0]), np.zeros(10000)), axis=0)
    print("new_train_images:", new_train_images)
    print("new_train_labels:", new_train_labels)
    model[i].fit(new_train_images, new_train_labels)
    print("Model", i, "trained")
    break

predictions = model.predict(test_images)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")

# Save the SVM model parameters to a file
filename = 'svm_model' + str(gamma) + '.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)