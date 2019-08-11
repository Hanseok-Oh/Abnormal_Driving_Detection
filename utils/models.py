from keras.models import Model, Sequential
import keras.layers as L
import utils

def ConvLSTM(optimizer):
    input_shape = (256, 256, 1)
    init_channel = 32
    block_num = 5
    drop_rate = 0.2

    input_1 = L.Input(shape=input_shape)
    input_2 = L.Input(shape=input_shape)
    input_3 = L.Input(shape=input_shape)
    weights_input = L.Input(shape=input_shape)

    encoder = Sequential(name='encoder')
    for i in range(block_num):
        if i == 0:
            encoder.add(L.Conv2D(init_channel*(i+1), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape))
        else:
            encoder.add(L.Conv2D(init_channel*(i+1), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'))
        encoder.add(L.Conv2D(init_channel*(i+1), (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))
        encoder.add(L.Dropout(rate=drop_rate))

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)
    encoded_3 = encoder(input_3)

    reshape = (1, *encoder.output_shape[1:])
    reshaped_1 = L.Reshape(reshape)(encoded_1)
    reshaped_2 = L.Reshape(reshape)(encoded_2)
    reshaped_3 = L.Reshape(reshape)(encoded_3)

    concat = L.Concatenate(axis=1)([reshaped_1, reshaped_2, reshaped_3])
    convlstm = L.ConvLSTM2D(init_channel*2, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=True)(concat)
    convlstm = L.ConvLSTM2D(init_channel*2, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=False)(convlstm)

    decoder_shape = (i.value for i in convlstm.get_shape()[1:])
    decoder = Sequential(name='decoder')

    for i in range(block_num):
        if i == 0:
            decoder.add(L.Conv2DTranspose(init_channel*(block_num-i), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=decoder_shape))
        else:
            decoder.add(L.Conv2DTranspose(init_channel*(block_num-i), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'))
        decoder.add(L.Conv2D(init_channel*(block_num-i), (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))
        decoder.add(L.Dropout(rate=drop_rate))
    decoder.add(L.Conv2D(1, (3,3), strides=1, activation='sigmoid', padding='same', kernel_initializer='he_normal'))

    output = decoder(convlstm)

    model = Model(inputs=[input_1, input_2, input_3, weights_input], outputs=output)
    model.compile(optimizer=optimizer, loss = utils.custom_loss(weights_input))
    print(model.summary())
    return model

'''
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
'''
