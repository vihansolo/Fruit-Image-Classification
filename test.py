"""
Image Classification Using Convolutional Neural Networks in Python (TensorFlow Keras)

train.py: main driver program which creates the CNN model
test.py: the python script to test the created CNN model on

@2020 Created by Vihang Garud
"""

# Importing necessary libraries
import cv2
import random
import glob
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Setting random seed to regenerate the results
random.seed(309)
np.random.seed(309)
tf.random.set_seed(309)


def import_images():
    """
    Importing the images and creating the datatset

    :return: features: the preprocessed images as the X for testing the model
             labels: the class labels used for testing the model
    """

    features = []
    labels = []

    # Querying images from all classes with intermediate pre-processing
    for file in glob.glob('test/cherry/*.jpg'):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = preprocess_images(image)
        image = img_to_array(image)
        features.append(image)
        labels.append(0)

    for file in glob.glob('test/strawberry/*.jpg'):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = preprocess_images(image)
        image = img_to_array(image)
        features.append(image)
        labels.append(1)

    for file in glob.glob('test/tomato/*.jpg'):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = preprocess_images(image)
        image = img_to_array(image)
        features.append(image)
        labels.append(2)

    # Converting the images to numpy array for the CNN model
    features = np.array(features).astype('float32')
    labels = np.array(labels)

    return features, labels


def preprocess_images(image):
    """
    Pre-processing the provided images for testing

    :param image: the raw image read in the import_images method
    :return: image: intermediately pre-processed image with following things applied-
                    1. Changing RGB to BGR as openCV is used. This maintains the original color scheme.
                    2. Resizing image to (100 x 100) for faster execution and better model performance.
                    3. Gaussian Blur for smoothening the images.
    """

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (100, 100))
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


if __name__ == '__main__':

    # Importing the testing images
    X, Y = import_images()

    # One Hot Encoding
    Y = to_categorical(Y, 3)

    # Loading the saved model
    model = load_model('model/model.h5')

    # Evaluating the model
    score = model.evaluate(X, Y, verbose=0, batch_size=10)

    # Printing the test results
    print('Test Accuracy = {:.4f}'.format(score[1] * 100), '%')
    print('Test Loss = {:.4f}'.format(score[0]))
