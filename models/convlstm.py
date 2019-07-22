from keras.models import Model
import keras.layers as L

def ConvLSTM(offset_x):
    inputs = L.Input((offset_x, 256, 256, 1), name='inputs')
    x = L.ConvLSTM2D(32, (3,3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform', return_sequences=True)(inputs)
    x = L.ConvLSTM2D(64, (3,3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform', return_sequences=True)(x)
    x = L.ConvLSTM2D(64, (3,3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform', return_sequences=True)(x)
    x = L.ConvLSTM2D(128, (3,3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform', return_sequences=False, name='encoded')(x)

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
