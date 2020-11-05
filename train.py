"""Image Classification Using Convolutional Neural Networks in Python (TensorFlow Keras)

train.py: main driver program which creates the CNN model
test.py: the python script to test the created CNN model on

@2020 Created by Vihang Garud
"""

# Importing necessary libraries
import cv2
import random
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Setting random seed to regenerate the results
random.seed(309)
np.random.seed(309)
tf.random.set_seed(309)


def import_images():
    """
    Importing the images and creating the datatset

    :return: features: the preprocessed images as the X for classification
             labels: the class labels used for classification
    """

    features = []
    labels = []

    # Querying images from all classes with intermediate pre-processing
    for file in glob.glob('Train_data/cherry/*.jpg'):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = preprocess_images(image)
        image = img_to_array(image)
        features.append(image)
        labels.append(0)

    for file in glob.glob('Train_data/strawberry/*.jpg'):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = preprocess_images(image)
        image = img_to_array(image)
        features.append(image)
        labels.append(1)

    for file in glob.glob('Train_data/tomato/*.jpg'):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = preprocess_images(image)
        image = img_to_array(image)
        features.append(image)
        labels.append(2)

    # Converting the images to numpy array for the CNN model
    features = np.array(features, dtype='float32')
    labels = np.array(labels)

    return features, labels


def preprocess_images(image):
    """
    Pre-processing the provided images

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


def func_cnn_model():
    """
    Training the CNN model
    1. Further pre-processing.
    2. Constructing the model and training it.

    :return: model: the trained CNN model
    """

    # The input shape of the images
    input_shape = X[0].shape

    # The CNN model
    model = Sequential([

        # Rescaling the images for faster execution
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=input_shape),

        # Further pre-processing
        layers.experimental.preprocessing.RandomZoom(0.2),
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        layers.experimental.preprocessing.RandomContrast(0.4),

        # Convolutional layer
        layers.Conv2D(50, 5, input_shape=input_shape, padding='same', activation='relu'),
        # Pooling layer
        layers.MaxPool2D(),

        # Convolutional layer
        layers.Conv2D(100, 9, padding='same', activation='relu'),
        # Pooling layer
        layers.MaxPool2D(),

        # Dense layer with Dropout and Flattening
        layers.Dense(256, input_shape=input_shape, activation='relu'),
        layers.Dropout(0.4),
        layers.Flatten(),

        # Final dense layer
        layers.Dense(128, activation='relu'),
        layers.Dense(3),

        # Applying final softmax activation layer
        layers.Activation('softmax')
    ])

    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

    # Printing the model summary
    print(model.summary())
    print(model.layers[0].input_shape)
    print(model.layers[0].output_shape)

    return model


def plot_acc_and_loss():
    """
    Plotting the accuracy and loss curves

    :return: this methods prints the accuracy and loss graph of train set versus validation set
    """

    # Setting the accuracy plot points
    train_acc = graph.history['accuracy']
    test_acc = graph.history['val_accuracy']

    epochs = range(len(train_acc))

    # Plotting the accuracy graphs
    plt.plot(epochs, train_acc, 'r', label='train_acc')
    plt.plot(epochs, test_acc, 'b', label='test_acc')
    plt.title('Train Accuracy vs Test Accuracy')
    plt.legend()
    plt.figure()

    # Setting the loss plot points
    train_loss = graph.history['loss']
    test_loss = graph.history['val_loss']

    # Plotting the loss points
    plt.plot(epochs, train_loss, 'r', label='train_loss')
    plt.plot(epochs, test_loss, 'b', label='test_loss')
    plt.title('Train Loss vs Test Loss')
    plt.legend()
    plt.figure()

    # Plotting the graphs
    plt.show()


if __name__ == '__main__':

    # Importing the images
    X, Y = import_images()

    # Splitting the dataset into train and test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.ravel(), test_size=0.1, shuffle=True, random_state=0)

    # One Hot Encoding
    Y_train = to_categorical(Y_train, 3)
    Y_test = to_categorical(Y_test, 3)

    # CNN model
    cnn_model = func_cnn_model()

    # Fitting the CNN model
    graph = cnn_model.fit(X_train, Y_train, batch_size=16, epochs=100, verbose=1, validation_data=(X_test, Y_test))

    # Evaluating the model
    score = cnn_model.evaluate(X_test, Y_test, verbose=0)

    # Printing the results
    print('Test Accuracy = {:.4f}'.format(score[1] * 100), '%')
    print('Test Loss = {:.4f}'.format(score[0]))

    # Saving the CNN model
    cnn_model.save('model/model_1.h5')

    # Plotting accuracy and loss
    plot_acc_and_loss()
