# Modules
import os
import cv2
import torch
import numpy as np

# Load class
class Load():
    def __init__(self, path, type='total'):
        self.path = path # dataset path
        self.videos = os.listdir(self.path) # video list
        self.type = type # 전체영상 or 특정영상 (total or specific)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # gpu

        self.frame_width = 320
        self.frame_height = 240
        self.frame_channel = 3

    # 영상 불러오기
    def cap_video(self, num):
        # video 불러오기
        self.current_video = self.videos[num] # 현재 처리 video
        current_video_path = self.path + self.current_video
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

    # 데이터셋 laod
    def load_dataset(self):
        for i in range(len(self.videos)):
            self.cap_video(i)
            if i == 0:
                dataset = self.current_video_frame
            else:
                dataset = np.concatenate((dataset, self.current_video_frame))
        self.dataset = dataset
        print('데이터로드 완료')

    # 데이터 전처리
    def convert_frame(self):

        temp_frame = self.current_video_frame
        temp_frame /= 255.0

        self.array_frame = temp_frame # array type

        # tensor type
        tensor_frame = torch.from_numpy(temp_frame).float().to(self.device) # tensor로 변환
        self.tensor_frame = tensor_frame.permute(0, 3, 1, 2) # pytorch img 순서로 변환
