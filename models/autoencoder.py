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


def Autoencoder_v2(drop_rate=0):
    input_img = L.Input(shape=(256, 256, 3))
    conv1 = L.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = L.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = L.MaxPooling2D((2, 2))(conv1)
    conv2 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = L.MaxPooling2D((2, 2))(conv2)
    conv3 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = L.MaxPooling2D((2, 2))(conv3)
    conv4 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = L.MaxPooling2D((2, 2))(conv4)
    conv5 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = L.MaxPooling2D((2, 2))(conv5)
    encoded = L.Flatten()(pool5)

    input_latent = L.Input(shape=(8*8*128,))
    reshaped = L.Reshape((8,8,128))(input_latent)
    up6 = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(reshaped)
    conv6 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = L.Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    up7 = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv6)
    conv7 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = L.Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    up8 = L.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(conv7)
    conv8 = L.Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = L.Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
    up9 = L.Conv2DTranspose(16, (2, 2), strides=2, padding='same')(conv8)
    conv9 = L.Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = L.Conv2D(16, (3, 3), activation='sigmoid', padding='same')(conv9)
    up10 = L.Conv2DTranspose(3, (2, 2), strides=2, padding='same')(conv9)
    conv10 = L.Conv2D(3, (3, 3), activation='relu', padding='same')(up10)
    decoded = L.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv10)

    encoder = Model(input_img, encoded, name='encoder')
    decoder = Model(input_latent, decoded, name='decoder')

    _encoded = encoder(input_img)
    _decoded = decoder(_encoded)
    autoencoder = Model(input_img, _decoded)
    return autoencoder

a = Autoencoder_v2()
print(a.summary())
