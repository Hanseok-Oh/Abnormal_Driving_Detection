import keras
from keras.models import Model
import keras.layers as L
import keras.backend as K
import numpy as np


def Autoencoder_v1(drop_rate=0):
    input_img = L.Input(shape=(240, 320, 3))

    x = L.Conv2D(32, (7, 7), activation='relu', padding='same')(input_img)
    x = L.Dropout(drop_rate)(x)
    x = L.MaxPooling2D((5, 5))(x)
    x = L.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.MaxPooling2D((2, 2))(x)
    encoded = L.Flatten()(x)

    input_latent = L.Input(shape=(6*8*128,))
    _x = L.Reshape((6,8,128))(input_latent)
    _x = L.Conv2D(128, (3, 3), activation='relu', padding='same')(_x)
    _x = L.Dropout(drop_rate)(_x)
    _x = L.UpSampling2D((5, 5))(_x)
    _x = L.Conv2D(128, (3, 3), activation='relu', padding='same')(_x)
    _x = L.Dropout(drop_rate)(_x)
    _x = L.UpSampling2D((2, 2))(_x)
    _x = L.Conv2D(64, (3, 3), activation='relu', padding='same')(_x)
    _x = L.Dropout(drop_rate)(_x)
    _x = L.UpSampling2D((2, 2))(_x)
    _x = L.Conv2D(32, (3, 3), activation='relu', padding='same')(_x)
    _x = L.Dropout(drop_rate)(_x)
    _x = L.UpSampling2D((2, 2))(_x)
    decoded = L.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(_x)

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_latent, decoded, name='decoder')

    input_total = L.Input(shape=(240, 320, 3))
    _encoded = encoder(input_total)
    _decoded = decoder(_encoded)
    autoencoder = Model(input_total, _decoded)

    return autoencoder


def Autoencoder_v2():
    input_img = L.Input(shape=(256, 256, 3))
    conv1 = L.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = L.MaxPooling2D((2, 2))(conv1)
    conv2 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = L.MaxPooling2D((2, 2))(conv2)
    conv3 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = L.MaxPooling2D((2, 2))(conv3)
    conv4 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = L.MaxPooling2D((2, 2))(conv4)
    conv5 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    pool5 = L.MaxPooling2D((2, 2))(conv5)
    conv6 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(pool5)
    pool6 = L.MaxPooling2D((2, 2))(conv6)
    encoded = L.Flatten()(pool6)

    input_latent = L.Input(shape=(4*4*256,))
    reshaped = L.Reshape((4,4,256))(input_latent)
    _up1 = L.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(reshaped)
    _conv1 = L.Conv2D(256, (3, 3), activation='relu', padding='same')(_up1)
    _up2 = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(_conv1)
    _conv2 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(_up2)
    _up3 = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(_conv2)
    _conv3 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(_up3)
    _up4 = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(_conv3)
    _conv4 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(_up4)
    _up5 = L.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(_conv4)
    _conv5 = L.Conv2D(32, (3, 3), activation='relu', padding='same')(_up5)
    _up6 = L.Conv2DTranspose(3, (2, 2), strides=2, padding='same')(_conv5)
    _conv6 = L.Conv2D(3, (3, 3), activation='relu', padding='same')(_up6)
    decoded = L.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(_conv6)

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_latent, decoded, name='decoder')

    _encoded = encoder(input_img)
    _decoded = decoder(_encoded)
    autoencoder = Model(input_img, _decoded)
    return autoencoder
