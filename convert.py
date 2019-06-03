# Modules
import os
import cv2
import torch
import numpy as np
import pandas as pd
import skvideo.io


#argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", help='load path', type=str, nargs='?', default='C:/Users/Yoon/Desktop/프로젝트/이상운전/data/frame/')
parser.add_argument("--save_path", help='save path', type=str, nargs='?', default='C:/Users/Yoon/Desktop/프로젝트/이상운전/data/array/')
parser.add_argument("--dataset_name", help='dataset name', type=str, nargs='?', default='cctv')

args = parser.parse_args()
load_path = args.load_path
save_path = args.save_path
dataset_name = args.dataset_name

# Load class
class Convert():
    def __init__(self, load_path=load_path, save_path=save_path, dataset_name=dataset_name):
        self.load_path = load_path
        self.save_path = save_path
        self.dataset_name = dataset_name

        self.videos = os.listdir(self.load_path) # video list

        # video size
        self.frame_width = 320
        self.frame_height = 240
        self.frame_channel = 3

    # 영상 불러오기
    def cap_video(self, num):
        # video 불러오기
        current_video = self.videos[num] # 현재 처리 video
        current_video_path = self.load_path + current_video
        cap = cv2.VideoCapture(current_video_path)

        # current video 정보
        self.current_video_frame_num = int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)) # frame 수

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

    # 영상 불러오기2
    def cap_video_2(self, num):
        current_video = self.videos[num] # 현재 처리 video
        current_video_path = self.load_path + current_video
        self.current_video_frame = skvideo.io.vread(current_video_path)

    # convert dataset
    def convert_dataset(self):
        for i in range(len(self.videos)):
            #self.cap_video(i)
            self.cap_video_2(i)
            if i == 0:
                dataset = self.current_video_frame
            else:
                dataset = np.concatenate((dataset, self.current_video_frame))

            if i % 10 == 0:
                print("{}% 변환 완료".format(str(int(((i+1) / len(self.videos)) * 100))))

        self.dataset = dataset

    # dataset info
    def dataset_info(self):
        days = [i.split('_')[0] for i in self.videos]
        times = [i.split('_')[1] for i in self.videos]
        names = [i.split('_')[2] for i in self.videos]
        self.datainfo = pd.DataFrame({'Day':days, 'Time':times, 'CCTV_Name':names})

    # save dataset
    def save_dataset(self):
        np.save(file=self.save_path + self.dataset_name + '.npy', arr=self.dataset)
        self.datainfo.to_csv(self.save_path + self.dataset_name + '.csv' ,encoding='utf-8')

if __name__ == '__main__':
    c = Convert()
    c.convert_dataset()
    c.dataset_info()
    c.save_dataset()
    print('변환 완료')
