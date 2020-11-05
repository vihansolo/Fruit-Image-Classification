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

random.seed(309)
np.random.seed(309)
tf.random.set_seed(309)


def import_images():

    features = []
    labels = []

    for file in glob.glob('Train_data/cherry/*.jpg'):
        image = cv2.imread(file)
        image = cv2.resize(image, (64, 64))
        image = img_to_array(image)
        features.append(image)
        labels.append(0)

    for file in glob.glob('Train_data/strawberry/*.jpg'):
        image = cv2.imread(file)
        image = cv2.resize(image, (64, 64))
        image = img_to_array(image)
        features.append(image)
        labels.append(1)

    for file in glob.glob('Train_data/tomato/*.jpg'):
        image = cv2.imread(file)
        image = cv2.resize(image, (64, 64))
        image = img_to_array(image)
        features.append(image)
        labels.append(2)

    features = np.array(features).astype('float32')
    features = features / 255
    labels = np.array(labels)

    return features, labels


def func_nn_model():

    input_shape = X[0].shape

    model = Sequential([

        layers.Dense(256, input_shape=input_shape, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Flatten(), layers.Dense(128, activation='relu'), layers.BatchNormalization(), layers.Dense(3),
        layers.Activation('softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    print(model.summary())
    print(model.layers[0].input_shape)
    print(model.layers[0].output_shape)

    return model


def plot_acc_and_loss(graph):

    train_acc = graph.history['accuracy']
    test_acc = graph.history['val_accuracy']

    epochs = range(len(train_acc))

    plt.plot(epochs, train_acc, 'r', label='train_acc')
    plt.plot(epochs, test_acc, 'b', label='test_acc')
    plt.title('Train Accuracy vs Test Accuracy')
    plt.legend()
    plt.figure()

    train_loss = graph.history['loss']
    test_loss = graph.history['val_loss']

    plt.plot(epochs, train_loss, 'r', label='train_loss')
    plt.plot(epochs, test_loss, 'b', label='test_loss')
    plt.title('Train Loss vs Test Loss')
    plt.legend()
    plt.figure()

    plt.show()


if __name__ == '__main__':

    X, Y = import_images()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.ravel(), test_size=0.3, shuffle=True, random_state=309)
    Y_train = to_categorical(Y_train, 3)
    Y_test = to_categorical(Y_test, 3)

    nn_model = func_nn_model()
    graph = nn_model.fit(X_train, Y_train, batch_size=10, epochs=100, verbose=1, validation_data=(X_test, Y_test))
    score = nn_model.evaluate(X_test, Y_test, verbose=0)

    print('Test Accuracy = {:.4f}'.format(score[1] * 100), '%')
    print('Test Loss = {:.4f}'.format(score[0]))

    nn_model.save('Models/base.h5')

    plot_acc_and_loss(graph)
