import keras
import tensorflow as tf
import scipy.stats as sp
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


def make_video(video_name, pred):
    video = cv2.VideoWriter('{}.avi'.format(video_name), 0, 12, (256, 256), False)

    for i in range(len(pred)):
        frame = pred[i][:,:,0] * 255
        frame = np.uint8(frame)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    print('저장 완료')
