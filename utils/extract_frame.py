import av
import os
import cv2
import argparse
import numpy as np


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', help='동영상 로드 경로', type=str)
parser.add_argument('--save_path', help='이미지 저장 경로', type=str)
parser.add_argument('--offset', help='이지미 저장 offset', type=int, default=1)
args = parser.parse_args()


def get_video_list(load_path):
    video_list = os.listdir(load_path)
    return video_list

def get_video_frame(video, load_path, save_path, offset):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    video_save_path = os.path.join(save_path, video[:-4])
    if not os.path.isdir(video_save_path):
        os.mkdir(video_save_path)

    container = av.open(os.path.join(load_path, video))
    frames = container.decode(video=0)
    i = 0
    for frame in frames:
        if frame.index % offset == 0:
            frame.to_image().save(os.path.join(video_save_path, '{}.png'.format(i)))
            i += 1
        else:
            continue

def main(args):
    video_list = get_video_list(args.load_path)
    i = 0
    for video in video_list:
        i += 1
        get_video_frame(video, args.load_path, args.save_path, args.offset)
        print(round(i / len(video_list), 2) * 100, '% 완료')
    print('전체 완료')

if __name__ == '__main__':
    main(args)
