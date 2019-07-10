import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class DataLoader:
    def __init__(self, directory, batch_size=32, batch_per_video=4):
        self.directory = directory
        self.batch_size = batch_size
        self.batch_per_video = batch_per_video

    def autoencoder_loader(self):
        while True:
            selected_video = self._choose_random_video()

            for i in range(len(selected_video)):
                video = selected_video[i]
                frame = self._choose_autoencoder_frame(video)

                if i == 0:
                    X = frame
                else:
                    X = np.concatenate((X, frame))

            X = X.astype(float) / 255
            yield (X, X)



    def rnn_loader(self, offset_x, offset_y):

        while True:
            selected_video = self._choose_random_video()

            for i in range(len(selected_video)):
                video = selected_video[i]
                frame_x, frame_y = self._choose_rnn_frame(video, offset_x, offset_y)


                if i == 0:
                    X = frame_x
                    Y = frame_y
                else:
                    X = np.concatenate((X, frame_x))
                    Y = np.concatenate((Y, frame_y))

            X = X.astype('float32') / 255
            Y = Y.astype('float32') / 255
            yield ([X[:,i,:,:,:] for i in range(offset_x)], Y)


    def _choose_random_video(self):
        video_list = os.listdir(self.directory)
        selected_index = np.random.randint(low=0, high=len(video_list), size=int(self.batch_size / self.batch_per_video))
        selected_video = [video_list[i] for i in selected_index]
        return selected_video


    def _choose_autoencoder_frame(self, video):
        video_path = os.path.join(self.directory, video)
        frame_list = os.listdir(video_path)

        selected_index = np.random.randint(0, len(frame_list), self.batch_per_video)
        selected_frame = np.array([idx_to_array(i, video_path) for i in selected_index])
        return selected_frame


    def _choose_rnn_frame(self, video, offset_x, offset_y):
        video_path = os.path.join(self.directory, video)
        frame_list = os.listdir(video_path)

        selected_index_y = np.random.randint(offset_y, len(frame_list), self.batch_per_video)
        selected_index_x = [[y - offset_y + x for x in range(offset_x)] for y in selected_index_y]
        selected_frame_y = np.array([idx_to_array(i, video_path) for i in selected_index_y])
        selected_frame_x = [[idx_to_array(i, video_path) for i in l] for l in selected_index_x]
        return selected_frame_x, selected_frame_y


def idx_to_array(idx, video_path):
    img_path = os.path.join(video_path, '{}.png'.format(idx))
    im = Image.open(img_path)
    arr = np.array(im)
    return arr


a = DataLoader('C:/users/yoon/downloads/UCSD/UCSDped1/Train/')
ae = a.autoencoder_loader()
print(next(ae))
