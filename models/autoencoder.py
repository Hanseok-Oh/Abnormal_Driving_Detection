import tensorflow as tf
import keras
from keras.models import Model
import keras.layers as L
import keras.backend as K
import numpy as np


def Autoencoder():
    input_img = L.Input(shape=(256, 256, 3), name='encoder_input')
    conv1 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = L.MaxPooling2D((2, 2))(conv1)
    conv2 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = L.MaxPooling2D((2, 2))(conv2)
    conv3 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = L.MaxPooling2D((2, 2))(conv3)
    conv4 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = L.MaxPooling2D((2, 2))(conv4)
    conv5 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    pool5 = L.MaxPooling2D((2, 2))(conv5)
    conv6 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(pool5)
    pool6 = L.MaxPooling2D((2, 2))(conv6)
    conv7 = L.Conv2D(512, (3, 3), activation='relu', padding='same')(pool6)
    pool7 = L.MaxPooling2D((2, 2))(conv7)
    encoded = L.Flatten(name='encoder_output')(pool7)

    input_latent = L.Input(shape=(2*2*512,), name='decoder_input')
    reshaped = L.Reshape((2,2,512))(input_latent)
    _up1 = L.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(reshaped)
    _conv1 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(_up1)
    _up2 = L.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(_conv1)
    _conv2 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(_up2)
    _up3 = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(_conv2)
    _conv3 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(_up3)
    _up4 = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(_conv3)
    _conv4 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(_up4)
    _up5 = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(_conv4)
    _conv5 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(_up5)
    _up6 = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(_conv5)
    _conv6 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(_up6)
    _up7 = L.Conv2DTranspose(3, (2, 2), strides=2, padding='same')(_conv6)
    decoded = L.Conv2D(3, (3, 3), activation='relu', padding='same', name='decoder_output')(_up7)

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_latent, decoded, name='decoder')

    _encoded = encoder(input_img)
    _decoded = decoder(_encoded)
    autoencoder = Model(input_img, _decoded)
    return autoencoder

def Autoencoder_test():
    input_img = L.Input(shape=(256, 256, 3), name='encoder_input')
    conv1 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = L.AveragePooling2D((2, 2))(conv1)
    conv2 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = L.AveragePooling2D((2, 2))(conv2)
    conv3 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = L.AveragePooling2D((2, 2))(conv3)
    conv4 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = L.AveragePooling2D((2, 2))(conv4)
    conv5 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    pool5 = L.AveragePooling2D((2, 2))(conv5)
    conv6 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(pool5)
    pool6 = L.AveragePooling2D((2, 2))(conv6)
    conv7 = L.Conv2D(512, (3, 3), activation='relu', padding='same')(pool6)
    pool7 = L.AveragePooling2D((2, 2))(conv7)
    encoded = L.Flatten(name='encoder_output')(pool7)

    input_latent = L.Input(shape=(2*2*512,), name='decoder_input')
    reshaped = L.Reshape((2,2,512))(input_latent)
    _up1 = L.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(reshaped)
    _conv1 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(_up1)
    _up2 = L.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(_conv1)
    _conv2 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(_up2)
    _up3 = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(_conv2)
    _conv3 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(_up3)
    _up4 = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(_conv3)
    _conv4 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(_up4)
    _up5 = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(_conv4)
    _conv5 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(_up5)
    _up6 = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(_conv5)
    _conv6 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(_up6)
    _up7 = L.Conv2DTranspose(3, (2, 2), strides=2, padding='same')(_conv6)
    decoded = L.Conv2D(3, (3, 3), activation='relu', padding='same', name='decoder_output')(_up7)

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_latent, decoded, name='decoder')

    _encoded = encoder(input_img)
    _decoded = decoder(_encoded)
    autoencoder = Model(input_img, _decoded)
    return autoencoder
