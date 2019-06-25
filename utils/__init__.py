import keras
import tensorflow as tf
from .dataloader import *

def ae_generator(dataloader, batch_size):
    while True:
        cctv = dataloader.choose_random_video()
        for _ in range(int(len(cctv) / batch_size)):
            random_idx = np.random.randint(0,len(cctv),batch_size)
            batch = cctv[random_idx]
            yield (batch, batch)


def rnn_generator(dataloader, encoder, batch_size, offset_x, offset_y, graph):
    with graph.as_default():
        while True:
            cctv = dataloader.choose_random_video()
            latent = encoder.predict(cctv)

            for _ in range(int(len(cctv) / batch_size)):
                idx_x = [[i+j for j in range(offset_x)] for i in range(len(cctv) - offset_y)]
                idx_y = [i for i in range(offset_y, len(cctv))]
                x = latent[idx_x]
                y = latent[idx_y]

                random_idx = np.random.randint(0,len(y),batch_size)
                batch_x = x[random_idx]
                batch_y = y[random_idx]

                yield (batch_x, batch_y)


def save_model(model, save_path):
    # save model
    model_json = model.to_json()
    with open('{}.json'.format(save_path), 'w') as json_file:
        json_file.write(model_json)
    # save weight
    model.save_weights('{}.h5'.format(save_path))
    print('저장 완료')


def load_model(load_path):
    json_file = open("{}.json".format(load_path), "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights('{}.h5'.format(load_path))
    return loaded_model
