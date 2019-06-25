import keras
from keras.models import Model
import keras.layers as L
import keras.backend as K
import numpy as np


def Autoencoder_v1(drop_rate=0.5):
    input_img = L.Input(shape=(240, 320, 3))

    x = L.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = L.Dropout(drop_rate)(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.MaxPooling2D((2,2))(x)
    encoded = L.Flatten(name='encoder')(x)

    x = L.Reshape((15,20,256))(encoded)
    x = L.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.UpSampling2D((2, 2))(x)
    x = L.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.UpSampling2D((2, 2))(x)
    x = L.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.UpSampling2D((2, 2))(x)
    x = L.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.UpSampling2D((2, 2))(x)
    decoded = L.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoder')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder


def Autoencoder_v2(drop_rate=0.5):
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


def Autoencoder_v3(drop_rate=0):
    input_img = L.Input(shape=(240, 320, 3))

    x = L.Conv2D(32, (7, 7), activation='relu', padding='same')(input_img)
    x = L.Dropout(drop_rate)(x)
    x = L.AveragePooling2D((5, 5))(x)
    x = L.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.AveragePooling2D((2, 2))(x)
    x = L.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.AveragePooling2D((2, 2))(x)
    x = L.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = L.Dropout(drop_rate)(x)
    x = L.AveragePooling2D((2, 2))(x)
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
