# Modules
import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class VideoLoad:
    def __init__(self, load_path):
        self.load_path = load_path

    # select video
    def select_video(self, random=True, video_idx=0):
        videos = os.listdir(self.load_path)

        if random:
            idx = np.random.choice(3)
        else:
            idx = video_idx

        self.selected_video = os.path.join(self.load_path, videos[idx])

    # video capture
    def cap_video(self):
        # video 불러오기
        cap = cv2.VideoCapture(self.selected_video)

        # current video 정보
        video_frame_num = int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)) # frame 수

        # capture
        video_frame = []
        for i in range(video_frame_num):
            ret, frame = cap.read()
            video_frame.append(frame)

            if cv2.waitKey(30) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        temp_video_frame = np.array(video_frame)
        self.video_frame = np.transpose(temp_video_frame, (0,3,1,2))

# custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, offset):
        self.data = data
        self.offset = offset

        # split x y
        self.x_idx = np.arange(len(data) - offset)
        self.y_idx = np.arange(offset, len(data))

        self.x_data = self.data[self.x_idx]
        self.y_data = self.data[self.y_idx]

        # data info
        self.height = 240
        self.width = 320
        self.channel = 3

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

# load data
def get_dataset(video_path, offset):
    videoloader = VideoLoad(video_path)
    videoloader.select_video()
    videoloader.cap_video()

    frame = videoloader.video_frame
    return CustomDataset(frame, offset)
