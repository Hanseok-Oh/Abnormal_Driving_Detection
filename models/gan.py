import tensorflow as tf
import keras
from keras.models import Model, Sequential
import keras.layers as L
import keras.backend as K
import numpy as np

class GAN:
    def __init__(self, autoencoder):
        self.input_shape = (256, 256, 3)
        self.G = self.generator(autoencoder)
        self.D = self.discriminator()
        self.AM = self.adversarial()

    def generator(self, autoencoder):
        g = autoencoder
        g.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4))
        g.summary()
        return g

    def discriminator(self):
        d = Sequential()
        d.add(L.Conv2D(64, (5, 5), strides=2, padding='same', input_shape=self.input_shape))
        d.add(L.LeakyReLU(0.2))
        d.add(L.Conv2D(128, (5, 5), strides=2, padding='same'))
        d.add(L.LeakyReLU(0.2))
        d.add(L.Conv2D(256, (5, 5), strides=2, padding='same'))
        d.add(L.LeakyReLU(0.2))
        d.add(L.Conv2D(512, (5, 5), strides=2, padding='same'))
        d.add(L.LeakyReLU(0.2))
        d.add(L.Flatten())
        d.add(L.Dense(128))
        d.add(L.Dense(1))
        d.add(L.Activation('sigmoid'))
        d.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4))
        d.summary()
        return d


    def adversarial(self):
        am = Sequential()

        self.G.name = 'generator'
        am.add(self.G)

        self.D.name = 'discriminator'
        self.D.trainable = False
        am.add(self.D)

        am.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4))
        am.summary()
        return am


    def train(self, x, y):
        valid = np.ones((len(x), 1))
        fake = np.zeros((len(x), 1))

        gen_x = self.G.predict(x)
        d_loss_real = self.D.train_on_batch(x, valid)
        d_loss_fake = self.D.train_on_batch(gen_x, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = self.AM.train_on_batch(x, valid)
        return d_loss, g_loss
