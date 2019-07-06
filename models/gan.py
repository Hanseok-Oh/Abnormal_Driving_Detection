import tensorflow as tf
import keras
from keras.models import Model, Sequential
import keras.layers as L
import keras.backend as K
import numpy as np
from autoencoder import Autoencoder

class GAN:
    def __init__(self, autoencoder):
        self.G = autoencoder
        self.D = None

    def discriminator(self):
        if self.D:
            return self.D
        input_shape = (256, 256, 3)
        self.D = Sequential()
        self.D.add(L.Conv2D(64, (5, 5), strides=2, input_shape=input_shape, padding='same'))
        self.D.add(L.LeakyReLU(0.2))
        self.D.add(L.Conv2D(128, (5, 5), strides=2, input_shape=input_shape, padding='same'))
        self.D.add(L.LeakyReLU(0.2))
        self.D.add(L.Conv2D(256, (5, 5), strides=2, input_shape=input_shape, padding='same'))
        self.D.add(L.LeakyReLU(0.2))
        self.D.add(L.Conv2D(512, (5, 5), strides=2, input_shape=input_shape, padding='same'))
        self.D.add(L.LeakyReLU(0.2))
        self.D.add(L.Flatten())
        self.D.add(L.Dense(1))
        self.D.add(L.Activation('sigmoid'))
        self.D.summary()
        return self.D

ae = Autoencoder(optimizer=keras.optimizers.SGD(lr=0.01))
g = GAN(ae)
print(g.G.summary())
