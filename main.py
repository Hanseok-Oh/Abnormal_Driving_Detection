# modules
import keras
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.dataloader import Dataloader
from models.autoencoder import Vanilla_Autoencoder


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', help='train or test', type=str, default='train')
parser.add_argument('--video_path', help='video directory', type=str)
parser.add_argument('--step', help='video steps', type=int, default=1000)
parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
args = parser.parse_args()

def get_generator(dataloader, batch_size):
    while True:
        cctv = dataloader.choose_random_video()
        for _ in range(int(len(cctv) / batch_size)):
            random_idx = np.random.randint(0,len(cctv),batch_size)
            batch = cctv[random_idx]
            yield (batch, batch)


def train(args):
    dataloader = Dataloader(args.video_path)
    #datagen = get_generator(dataloader, args.batch_size)
    model = Vanilla_Autoencoder()
    print(model.summary())

    print('학습 시작')
    for st in range(args.step):
        cctv = dataloader.choose_random_video()
        for _ in range(int(len(cctv) / args.batch_size)):
            random_idx = np.random.randint(0,len(cctv), args.batch_size)
            batch = cctv[random_idx]
            history = model.train_on_batch(batch, batch)

        if st % 100 == 0:
            img = model.predict(cctv)
            fig = plt.figure()
            print(plt.imshow(img[0]))

    print('학습 완료')
    print(history)

def test(args):
    print('test')

def main(args):
    if args.train == 'train':
        train(args)
    else:
        test(args)

if __name__ == '__main__':
    main(args)
