import argparse
import pandas as pd
import matplotlib.pyplot as plt

import keras
import keras.layers as L
import keras.backend as K
from keras.callbacks import ModelCheckpoint

import utils
from utils.models import ConvLSTM
from utils.data import Dataset

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='train')
parser.add_argument('--directory', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--steps_per_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_per_video', type=int, default=1)
parser.add_argument('--offset_x', nargs='+', type=int, default=[1, 7, 15])
parser.add_argument('--offset_y', type=int, default=30)
parser.add_argument('--init_channel', type=int, default=32)
parser.add_argument('--block_num', type=int, default=3)
args = parser.parse_args()


def train(dataloader, model, epochs, steps_per_epoch, save_path):
    checkpoint = ModelCheckpoint('{}.h5'.format(save_path), monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = model.fit_generator(
        generator=dataloader,
        epochs = epochs,
        steps_per_epoch= steps_per_epoch,
        callbacks=callbacks_list
    )
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('{}.csv'.format(save_path))


def main(args):
    dataset = Dataset(args.directory, args.offset_x, args.offset_y, args.batch_size, args.batch_per_video)

    if args.train == 'train':
        dataloader = dataset.train_loader()
        optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-4)
        model = ConvLSTM(optimizer, args.init_channel, args.block_num)
        train(dataloader, model, args.epochs, args.steps_per_epoch, args.save_path)

    else:
        x, y = dataset.test_loader(0)
        model = ConvLSTM(optimizer, args.init_channel, args.block_num)
        model = utils.load_model(model, args.save_path)
        pred = model.predict(x)
        utils.make_video('predict', pred)

if __name__ == '__main__':
    main(args)
