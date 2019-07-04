import argparse

import utils
from utils import dataloader
from models import autoencoder, recurrent

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='데이터 디렉토리', type=str)
parser.add_argument('--train', help='train or test', type=str, defualt='train')
parser.add_argument('--model', help='model', typ=str, default='autoencoder')
parser.add_argument('--batch_size', help='batch size', type=int, default=32)
parser.add_argument('--batch_per_video', help='batch per video', type=str, default=4)
parser.add_argument('--model_path', help='model save/load path', type=str)
args = parser.parse_args()


def train():
    return

def test():
    return

def main(args):
    train_loader = dataloader.DataLoader(args.data_path, batch_size=args.batch_size, batch_per_video=args.batch_per_video)

    model_ae = autoencoder.Autoencoder()
    rnn_dim = ae.get_layer('encoder').output_shape[1]
    model_rnn = recurrent.RNN(model_ae, 3, rnn_dim)


if __name__ == '__main__':
    main(args)
