import os
import h5py
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras

# Loads the dataset - here Normalized MNIST
def loadDataset():
    from tensorflow.python.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape([-1,28,28,1])/255.
    X_test = X_test.reshape([-1,28,28,1])/255.
    print(X_train.shape)
    print(X_test.shape)
    return X_train, X_test
