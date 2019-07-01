import keras
import tensorflow as tf
from .dataloader import *


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
