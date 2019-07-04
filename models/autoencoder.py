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

def VAE(latent_dim=256):
    encoder_input = L.Input(shape=(256, 256, 3), name='encoder_input')
    en = L.Conv2D(64, (3, 3), padding='same')(encoder_input)
    en = L.BatchNormalization()(en)
    en = L.Activation('relu')(en)
    en = L.MaxPooling2D((2, 2))(en)
    en = L.Conv2D(64, (3, 3), padding='same')(en)
    en = L.BatchNormalization()(en)
    en = L.Activation('relu')(en)
    en = L.MaxPooling2D((2, 2))(en)
    en = L.Conv2D(128, (3, 3), padding='same')(en)
    en = L.BatchNormalization()(en)
    en = L.Activation('relu')(en)
    en = L.MaxPooling2D((2, 2))(en)
    en = L.Conv2D(128, (3, 3), padding='same')(en)
    en = L.BatchNormalization()(en)
    en = L.Activation('relu')(en)
    en = L.MaxPooling2D((2, 2))(en)
    en = L.Conv2D(256, (3, 3), padding='same')(en)
    en = L.BatchNormalization()(en)
    en = L.Activation('relu')(en)
    en = L.MaxPooling2D((2, 2))(en)
    en = L.Conv2D(512, (3, 3), padding='same')(en)
    en = L.BatchNormalization()(en)
    en = L.Activation('relu')(en)
    en = L.MaxPooling2D((2, 2))(en)
    en = L.Conv2D(512, (3, 3), padding='same')(en)
    en = L.BatchNormalization()(en)
    en = L.Activation('relu')(en)
    en = L.MaxPooling2D((2, 2))(en)

    unflatted_shape = K.int_shape(en)
    flatted= L.Flatten(name='encoder_output')(en)

    z_mean = L.Dense(latent_dim, name='z_mean')(flatted)
    z_log_var = L.Dense(latent_dim, name='z_log_var')(flatted)
    z = L.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    decoder_input = L.Input(shape=(latent_dim, ), name='decoder_input')
    de = L.Dense(np.prod(unflatted_shape[1:]))(decoder_input)
    de = L.Reshape((unflatted_shape[1:]))(de)
    de = L.Conv2DTranspose(512, (2, 2), strides=2, padding='same')(de)
    de = L.Conv2D(512, (3, 3), padding='same')(de)
    de = L.BatchNormalization()(de)
    de = L.Activation('relu')(de)
    de = L.Conv2DTranspose(512, (2, 2), strides=2, padding='same')(de)
    de = L.Conv2D(512, (3, 3), padding='same')(de)
    de = L.BatchNormalization()(de)
    de = L.Activation('relu')(de)
    de = L.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(de)
    de = L.Conv2D(256, (3, 3), padding='same')(de)
    de = L.BatchNormalization()(de)
    de = L.Activation('relu')(de)
    de = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(de)
    de = L.Conv2D(128, (3, 3), padding='same')(de)
    de = L.BatchNormalization()(de)
    de = L.Activation('relu')(de)
    de = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(de)
    de = L.Conv2D(128, (3, 3), padding='same')(de)
    de = L.BatchNormalization()(de)
    de = L.Activation('relu')(de)
    de = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(de)
    de = L.Conv2D(64, (3, 3), padding='same')(de)
    de = L.BatchNormalization()(de)
    de = L.Activation('relu')(de)
    de = L.Conv2DTranspose(3, (2, 2), strides=2, padding='same')(de)
    de = L.Conv2D(3, (3, 3), padding='same')(de)
    de = L.BatchNormalization()(de)
    decoded = L.Activation('sigmoid')(de)

    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_input, decoded, name='decoder')
    output = decoder(encoder(encoder_input)[2])

    def vae_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.mse(y_true, y_pred)
        kl_loss = 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    vae = Model(encoder_input, output)
    vae.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=vae_loss)
        
    return vae

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
