import tensorflow as tf
import keras
from keras.models import Model, Sequential
import keras.layers as L
import keras.backend as K
import numpy as np

def create_conv_block(model, size, init=False, input_shape=None, last=False):
    if init:
        model.add(L.Conv2D(size, (3, 3), strides=1, padding='same', input_shape=input_shape))
    else:
        model.add(L.Conv2D(size, (3, 3), strides=1, padding='same'))

    if last:
        model.add(L.Activation('sigmoid'))
    else:
        model.add(L.Activation('relu'))
    return model



def AutoEncoder(init_kernel_size=64, depth=5, input_shape = (128, 128, 3)):
    encoder = Sequential(name='encoder')

    for i in range(depth):
        kernel_size = init_kernel_size * (2 ** i)
        if i == 0:
            encoder = create_conv_block(encoder, kernel_size, init=True, input_shape=input_shape)
        else:
            encoder = create_conv_block(encoder, kernel_size)
        encoder = create_conv_block(encoder, kernel_size)
        #encoder.add(L.MaxPooling2D((2,2)))
        encoder.add(L.Conv2D(kernel_size, (3,3), strides=2, padding='same')) # strided conv

    encoder.add(L.Flatten())

    unflattened_shape = encoder.get_layer(index=-2).output_shape[1:]
    flattened_shape = encoder.get_layer(index=-1).output_shape[1:]

    decoder = Sequential(name='decoder')
    decoder.add(L.Reshape(target_shape=unflattened_shape, input_shape=flattened_shape))

    for i in reversed(range(depth)):
        kernel_size = init_kernel_size * (2 ** i)
        #decoder.add(L.UpSampling2D((2, 2)))
        decoder.add(L.Conv2DTranspose(kernel_size, (2, 2), strides=2, padding='same')) # upsampling
        decoder = create_conv_block(decoder, kernel_size)

        if i == 0:
            decoder = create_conv_block(decoder, 3, last=True)

    autoencoder = Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)
    return autoencoder


def VAE(optimizer, latent_dim=512):
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
    z = L.Lambda(_sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

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
    decoded = L.Activation('sigmoid')(de)

    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_input, decoded, name='decoder')
    output = decoder(encoder(encoder_input)[2])

    def vae_loss(y_true, y_pred, z_mean=z_mean, z_log_var=z_log_var):
        reconstruction_loss = (256*256) * K.mean(keras.losses.mse(y_true, y_pred), axis=[1,2])
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    vae = Model(encoder_input, output)
    vae.compile(
        optimizer=optimizer,
        loss=vae_loss)

    return vae

def _sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
