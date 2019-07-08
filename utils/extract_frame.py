import os
import cv2
import argparse
import numpy as np
from PIL import Image


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', help='동영상 로드 경로', type=str)
parser.add_argument('--save_path', help='이미지 저장 경로', type=str)
parser.add_argument('--size', help='이미지 저장 사이즈', type=int, default=128)
parser.add_argument('--split_ratio', help='train split ratio', type=float, default=0.7)
args = parser.parse_args()


def get_video_list(load_path):
    video_list = os.listdir(load_path)
    return video_list

def make_dir(save_path):
    save_path = os.path.abspath(save_path)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(os.path.join(save_path, 'train')):
        os.mkdir(os.path.join(save_path, 'train'))
    if not os.path.isdir(os.path.join(save_path, 'test')):
        os.mkdir(os.path.join(save_path, 'test'))

def get_video_frame(video, load_path, save_path, split_ratio):
    save_path = os.path.abspath(save_path)

    if np.random.rand() < split_ratio:
        video_save_path = os.path.join(save_path, 'train', video[:-4])
    else:
        video_save_path = os.path.join(save_path, 'test', video[:-4])

    video_save_path = os.path.abspath(video_save_path)

    if not os.path.isdir(video_save_path):
        os.mkdir(video_save_path)

    cap = cv2.VideoCapture(os.path.join(load_path, video))
    frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_len):
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(args.size, args.size), interpolation=cv2.INTER_LINEAR)
        os.chdir(video_save_path)
        cv2.imwrite('{}.png'.format(i), frame)
    cap.release()

def main(args):
    video_list = get_video_list(args.load_path)
    make_dir(args.save_path)

    i = 0
    for video in video_list:
        i += 1
        get_video_frame(video, args.load_path, args.save_path, args.split_ratio)

        if i % 100 == 0:
            print('{}% 완료'.format(int((i / len(video_list)) * 100)))
    print('전체 완료')

if __name__ == '__main__':
    main(args)
