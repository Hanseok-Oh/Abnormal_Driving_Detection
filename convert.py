# Modules
import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", help='비디오 경로', type=str, nargs='?', default='C:/Users/Yoon/Desktop/프로젝트/이상운전/data/frame/')
parser.add_argument("--array_path", help='array 경로', type=str, nargs='?', default='C:/Users/Yoon/Desktop/프로젝트/이상운전/data/array/')
parser.add_argument("--dataset_num", help='dataset order', type=str)

args = parser.parse_args()
video_path = args.video_path
array_path = args.array_path

# Load class
class Load():
    def __init__(self, type='total'):
        self.video_path = args.video_path # dataset path
        self.videos = os.listdir(self.video_path) # video list
        self.type = type # 전체영상 or 특정영상 (total or specific)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # gpu

        self.frame_width = 320
        self.frame_height = 240
        self.frame_channel = 3

    # 영상 불러오기
    def cap_video(self, num):
        # video 불러오기
        self.current_video = self.videos[num] # 현재 처리 video
        current_video_path = self.video_path + self.current_video
        cap = cv2.VideoCapture(current_video_path)

        # video 정보
        self.frame_num = int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)) # frame 수

        # capture
        current_video_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_video_frame.append(frame) # frame을 video_frame에 저장

            if cv2.waitKey(30) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.current_video_frame = np.array(current_video_frame, dtype=np.float32) # 현재 처리 video frame

    # load dataset
    def load_dataset(self):
        for i in range(len(self.videos)):
            self.cap_video(i)
            if i == 0:
                dataset = self.current_video_frame
            else:
                dataset = np.concatenate((dataset, self.current_video_frame))
        self.dataset = dataset

        self.datainfo = pd.DataFrame({'filename':self.videos})

    # save dataset
    def save_dataset(self):
        data_name = 'dataset_{}.npy'.format(args.dataset_num)
        data_info_name = 'datainfo_{}.csv'.format(args.dataset_num)
        np.save(args.array_path + name, self.dataset)
        self.datainfo.to_csv(args.array_path + data_info_name, encoding='utf-8')

if __name__ == '__main__':
    l = Load()
    l.load_dataset()
    l.save_dataset()
    print('변환 완료')
