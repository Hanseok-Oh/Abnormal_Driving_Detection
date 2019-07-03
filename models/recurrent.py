from keras.models import Model
import keras.layers as L

def RNN(autoencoder, offset_x, rnn_dim, autoencoder_trainable=False):
    encoder = autoencoder.get_layer('encoder')
    decoder = autoencoder.get_layer('decoder')
    encoder.trainable = autoencoder_trainable
    decoder.trainable = autoencoder_trainable

    input1 = L.Input((256,256,3))
    input2 = L.Input((256,256,3))
    input3 = L.Input((256,256,3))

    encoded1 = encoder(input1)
    encoded2 = encoder(input2)
    encoded3 = encoder(input3)

    encoded1 = L.Reshape((1,-1))(encoded1)
    encoded2 = L.Reshape((1,-1))(encoded2)
    encoded3 = L.Reshape((1,-1))(encoded3)

    concat = L.Concatenate(axis=1)([encoded1, encoded2, encoded3])
    lstm1 = L.LSTM(512, input_shape=(offset_x, rnn_dim))(concat)
    dense1 = L.Dense(rnn_dim)(lstm1)

    decoded = decoder(dense1)
    rnn = Model([input1, input2, input3], decoded)
    return rnn
