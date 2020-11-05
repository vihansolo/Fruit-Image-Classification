import cv2
import random
import glob
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


def func_cnn_model(train_x, train_y):

    input_shape = X[0].shape

    model = Sequential([

        layers.Conv2D(50, 5, input_shape=input_shape, padding='same', activation='relu'),
        layers.MaxPool2D(),

        layers.Conv2D(100, 9, padding='same', activation='relu'),
        layers.MaxPool2D(),

        layers.Dense(256, input_shape=input_shape, activation='relu'),
        layers.Dropout(0.4),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(3),

        layers.Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.fit(train_x, train_y, batch_size=10, epochs=100, verbose=0)

    return model


def ensemble_predictions(no_of_models, test_x):
    predictions = [model.predict(test_x) for model in no_of_models]
    predictions = np.array(predictions)

    return np.argmax(np.sum(predictions, axis=0), axis=1)


def evaluate_models(all_models, no_of_models, test_x, test_y):

    subset = all_models[:no_of_models]
    prediction = ensemble_predictions(subset, test_x)

    return accuracy_score(test_y, prediction)


if __name__ == '__main__':

    X, Y = import_images()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.ravel(), test_size=0.3, shuffle=True, random_state=309)
    Y_train = to_categorical(Y_train, 3)
    Y_test = to_categorical(Y_test, 3)

    n_models = 20
    scores = []

    models = [func_cnn_model(X_train, Y_train) for _ in range(n_models)]

    for i in range(n_models):
        score = evaluate_models(models, i, X_test, Y_test)
        print('Score : {:.3f}'.format(score))
        scores.append(score)
