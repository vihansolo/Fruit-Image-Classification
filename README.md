# Fruit-Image-Classification
 This is an Image Classification Project performed using Convolutional Neural Networks (CNNs) in Tensorflow Keras libraries. The main objective of the classification is to create a model which categorizes the images into three categories of fruits - cherry, strawberry, and tomato.

## Requirements and running the program 

1. The required python libraries to run this project are:
    - python-opencv
    - numpy
    - matplotlib
    - scikit-learn
    - tensorflow

2. Tensorflow GPU has been used to run the following script with Nvidia GTX 1050ti graphics card and following recommended versions -

    - Tensorflow-gpu 2.3.0
    - Nvidia CUDA 10.1
    - Nvidia cuDNN 7.6.4

3. To create a model:

    - Run the train script using "python train.py"
        - Creates a model inside "model/"

    - Run the test script using "python test.py" to check the classification on the test images.
    
## Model

The following model structure has been used to classify the images with partial preprocessing of the images performed in the CNN model itself -

| Layer (type) | Output Shape | Param # | 
|--------------|--------------|---------|
| rescaling (Rescaling) | (None, 100, 100, 3) | 0 |
| random_zoom (RandomZoom) | (None, 100, 100, 3) | 0 |
| random_flip (RandomFlip) | (None, 100, 100, 3) | 0 |
| random_contrast (RandomContrast) | (None, 100, 100, 3) | 0 |
| conv2d (Conv2D) | (None, 100, 100, 50) | 3800 |
| conv2d_1 (Conv2D) | (None, 100, 100, 75) | 93825|
| max_pooling2d (MaxPooling2D) | (None, 50, 50, 75) | 0 |
| conv2d_2 (Conv2D) | (None, 50, 50, 100) | 367600 |
| conv2d_3 (Conv2D) | (None, 50, 50, 125) | 612625 |
| max_pooling2d_1 (MaxPooling2D) | (None, 25, 25, 125) | 0 |
| dense (Dense) | (None, 25, 25, 256) | 32256 |
| dropout (Dropout) | (None, 25, 25, 256) | 0 |
| flatten (Flatten) | (None, 160000) | 0 |
| dense_1 (Dense) | (None, 128) | 20480128 |
| dense_2 (Dense) | (None, 3) | 387 |
| activation (Activation) | (None, 3) | 0 |

### Accuracy
The existing model accuracy is 73.33 %.
