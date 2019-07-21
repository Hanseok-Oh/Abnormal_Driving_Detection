from keras.models import Model
import keras.layers as L

def ConvLSTM():
    inputs = L.Input((3, 256, 256, 1))
    x = L.ConvLSTM2D(32, (3,3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform', return_sequences=True)(inputs)
    x = L.ConvLSTM2D(64, (3,3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform', return_sequences=True)(x)
    x = L.ConvLSTM2D(64, (3,3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform', return_sequences=True)(x)
    x = L.ConvLSTM2D(128, (3,3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform', return_sequences=False)(x)

    x = L.Conv2DTranspose(128, (2, 2), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = L.Conv2D(128, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = L.Conv2D(64, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = L.Conv2DTranspose(64, (2, 2), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = L.Conv2D(64, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = L.Conv2DTranspose(32, (2, 2), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = L.Conv2D(32, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = L.Conv2D(1, (3,3), strides=1, padding='same', activation='sigmoid', kernel_initializer='he_uniform')(x)

    model = Model(inputs, x)
    return model
