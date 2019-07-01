import av
import os
import cv2
import argparse
import numpy as np
from PIL import Image


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', help='동영상 로드 경로', type=str)
parser.add_argument('--save_path', help='이미지 저장 경로', type=str)
args = parser.parse_args()


def get_video_list(load_path):
    video_list = os.listdir(load_path)
    return video_list


def get_video_frame(video, load_path, save_path):
    save_path = os.path.abspath(save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    video_save_path = os.path.join(save_path, video[:-4])
    video_save_path = os.path.abspath(video_save_path)
    if not os.path.isdir(video_save_path):
        os.mkdir(video_save_path)

    cap = cv2.VideoCapture(os.path.join(load_path, video))
    frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_len):
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        os.chdir(video_save_path)
        cv2.imwrite('{}.png'.format(i), frame)
    cap.release()

def main(args):
    video_list = get_video_list(args.load_path)
    i = 0
    for video in video_list:
        i += 1
        get_video_frame(video, args.load_path, args.save_path)
        print(round(i / len(video_list), 2) * 100, '% 완료')
    print('전체 완료')

if __name__ == '__main__':
    main(args)


'''
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
            frame = cv2.resize(frame.to_ndarray(), dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
            Image.fromarray(frame).save(os.path.join(video_save_path, '{}.png'.format(i)))
            i += 1
        else:
            continue
'''
