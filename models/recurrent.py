import keras.layers as L

def RNN_v1(offset_x, rnn_input):
    rnn = L.Sequential()
    rnn.add(L.LSTM(512, input_shape=(offset_x, rnn_input)))
    rnn.add(L.Dense(rnn_input))
    return rnn

def RNN_v2(offset_x, input_shape):
    rnn = L.Sequential()
    rnn.add(L.LSTM(512, input_shape=(offset_x, input_shape), return_sequences=True))
    rnn.add(L.LSTM(512))
    rnn.add(L.Dense(6144))
    return rnn
