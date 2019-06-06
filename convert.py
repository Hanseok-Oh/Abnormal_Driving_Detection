# Modules
import os
import cv2
import torch
import numpy as np
import pandas as pd

#argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", help='load path', type=str)
parser.add_argument("--save_path", help='save path', type=str)

args = parser.parse_args()
load_path = args.load_path
save_path = args.save_path

# Convert class
class Convert():
    def __init__(self, load_path=load_path, save_path=save_path):
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
        for i in range(self.current_video_frame_num):
            ret, frame = cap.read()

            img_name = "{}_{}.png".format(num, i)
            os.chdir(save_path) # save path로 이동
            cv2.imwrite(img_name, frame) # frame to img

            if cv2.waitKey(30) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    # convert dataset
    def convert_dataset(self):
        for i in range(len(self.videos)):
            self.cap_video(i)

    # dataset info
    def dataset_info(self):
        days = [i.split('_')[0] for i in self.videos]
        times = [i.split('_')[1] for i in self.videos]
        names = [i.split('_')[2] for i in self.videos]
        self.datainfo = pd.DataFrame({'Day':days, 'Time':times, 'CCTV_Name':names})
        self.datainfo.to_csv(self.save_path + 'data_info.csv', encoding='utf-8')

if __name__ == '__main__':
    c = Convert()
    c.convert_dataset()
    c.dataset_info()
    print('변환 완료')
