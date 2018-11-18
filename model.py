import os
import sys
import h5py
import cv2

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import norm
from sklearn import manifold

from tensorflow.python.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.python.keras.layers import Convolution2D, UpSampling2D, MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.advanced_activations import ELU
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from config import latent_dim, imageSize

# Convolutional models
# x is input, z is 
def getModels():
    input_img = Input(shape=(imageSize, imageSize, 1))
    x = Convolution2D(32, (3, 3), padding='same', input_shape=(imageSize, imageSize, 1))(input_img)
    x = ELU()(x)
    x = MaxPooling2D(padding='same')(x)

    x = Convolution2D(64, (3, 3), padding='same')(x)
    x = ELU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Latent space // bottleneck layer
    x = Flatten()(x)
    x = Dense(latent_dim)(x)
    z = ELU()(x)

    ##### MODEL 1: ENCODER #####
    encoder = Model(input_img, z)
   
    # We instantiate these layers separately so as to reuse them for the decoder
    # Dense from latent space to image dimension
    x_decoded_dense1 = Dense(7 * 7 * 64)

    # Reshape for image
    x_decoded_reshape0 = Reshape((7, 7, 64))
    x_decoded_upsample0 = UpSampling2D((2, 2))
    x_decoded_conv0  = Convolution2D(32, (3, 3), padding='same')

    x_decoded_upsample3 = UpSampling2D((2, 2))
    x_decoded_conv3 = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')

    # Create second part of autoencoder
    x_decoded = x_decoded_dense1(z)
    x_decoded = ELU()(x_decoded)

    x_decoded = x_decoded_reshape0(x_decoded)
    x_decoded = x_decoded_upsample0(x_decoded)
    x_decoded = x_decoded_conv0(x_decoded)
    x_decoded = ELU()(x_decoded)

    # Tanh layer
    x_decoded = x_decoded_upsample3(x_decoded)
    decoded_img = x_decoded_conv3(x_decoded)

    ##### MODEL 2: AUTO-ENCODER #####
    autoencoder = Model(input_img, decoded_img)

    # Create decoder
    input_z = Input(shape=(latent_dim,))
    x_decoded_decoder = x_decoded_dense1(input_z)
    x_decoded_decoder = ELU()(x_decoded_decoder)

    x_decoded_decoder = x_decoded_reshape0(x_decoded_decoder)
    x_decoded_decoder = x_decoded_upsample0(x_decoded_decoder)
    x_decoded_decoder = x_decoded_conv0(x_decoded_decoder)
    x_decoded_decoder = ELU()(x_decoded_decoder)

    # Tanh layer
    x_decoded_decoder = x_decoded_upsample3(x_decoded_decoder)
    decoded_decoder_img = x_decoded_conv3(x_decoded_decoder)

    ##### MODEL 3: DECODER #####
    decoder = Model(input_z, decoded_decoder_img)
    return autoencoder, encoder, decoder

