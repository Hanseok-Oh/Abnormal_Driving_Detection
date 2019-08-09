import os
import cv2
import tkinter
import numpy as np
from tqdm import tqdm
from PIL import Image
from tkinter.filedialog import askdirectory


def get_path():
    root=tkinter.Tk()
    root.withdraw()
    load_path = askdirectory(title='load path')
    save_path = askdirectory(title='save path')
    root.destroy()
    return load_path, save_path

def get_video_list(load_path):
    video_list = os.listdir(load_path)
    return video_list

def make_dir(save_path):
    save_path = os.path.abspath(save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

def get_video_frame(video, load_path, save_path):
    save_path = os.path.abspath(save_path)
    video_save_path = os.path.join(save_path, video[:-4])
    video_save_path = os.path.abspath(video_save_path)

    if not os.path.isdir(video_save_path):
        os.mkdir(video_save_path)

    cap = cv2.VideoCapture(os.path.join(load_path, video))
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_len):
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(frame)
        os.chdir(video_save_path)
        cv2.imwrite('{}.png'.format(i), gray)
        cv2.imwrite('{}_fgbg.png'.format(i), fgmask)
    cap.release()

def main():
    load_path, save_path = get_path()
    video_list = get_video_list(load_path)
    make_dir(save_path)
    for i in tqdm(range(len(video_list)), mininterval=1):
        video = video_list[i]
        get_video_frame(video, load_path, save_path)

    print('전체 완료')


if __name__ == '__main__':
    main()
