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
            for video in selected_video:
                frame = self._choose_autoencoder_frame(video)
                if not X:
                    X = frame
                else:
                    X = np.concatenate((X, frame))

            X = X.astype(float) / 255
            yield (X, X)


    def rnn_loader(self, graph, encoder, offset_x, offset_y):
        with graph.as_default():
            while True:
                selected_video = self._choose_random_video()
                for video in selected_video:
                    frame_x, frame_y = self._choose_rnn_frame(video, offset_x, offset_y)
                    latent_x = np.array([encoder.predict(i) for i in frame_x])
                    latent_y = encoder.predict(frame_y)

                    if not X:
                        X = latent_x
                        Y = latent_y
                    else:
                        X = np.concatenate((X, latent_x))
                        Y = np.concatenate((Y, latent_y))

                X = X.astype('float32') / 255
                Y = Y.astype('float32') / 255
                yield (X, Y)


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
        selected_frame_x = np.array([[idx_to_array(i, video_path) for i in l] for l in selected_index_x])
        return selected_frame_x, selected_frame_y


def img_to_array(img):
    im = Image.open(img)
    arr = np.array(im)
    return arr

def idx_to_array(idx, video_path):
    img_path = os.path.join(video_path, '{}.png'.format(idx))
    im = Image.open(img_path)
    arr = np.array(im)
    return arr
