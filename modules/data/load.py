# Modules
import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class VideoLoad:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_list = os.listdir(self.video_path)

    # select video
    def select_video(self, random=True, video_idx=0):
        # index 추출
        if random:
            idx = np.random.choice(3)
        else:
            idx = video_idx
        self.selected_video = os.path.join(self.video_path, self.video_list[idx]) # video 선택

    # video capture
    def cap_video(self):
        cap = cv2.VideoCapture(self.selected_video) # video 불러오기
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
        self.video_frame_numpy = np.array(video_frame)
        self.video_frame_pytorch = np.transpose(self.video_frame_numpy, (0,3,1,2)) # pytorch 순서로 변경

# custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, offset, transform=None):
        self.data = data
        self.offset = offset # x,y frame 차이
        self.transform = transform

        # data info
        self.height = 240
        self.width = 320
        self.channel = 3

        # split x y
        self.x_idx = np.arange(len(data) - offset)
        self.y_idx = np.arange(offset, len(data))
        self.x_data = self.data[self.x_idx]
        self.y_data = self.data[self.y_idx]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        else:
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])

        return x, y

# load data
def load_dataset(video_path, batch_size, offset, transform):
    # video capture
    videoloader = VideoLoad(video_path)
    videoloader.select_video()
    videoloader.cap_video()

    dataset = CustomDataset(videoloader.video_frame_pytorch, offset, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader
