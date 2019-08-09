import argparse
import keras
import keras.layers as L
import keras.backend as K


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

args = parser.parse_args()


def train(dataloader, model, epochs, steps_per_epoch):
    hist = model.fit_generator(
        generator=dataloader,
        epochs = epochs,
        steps_per_epoch= steps_per_epoch
    )
    return

def test():
    return

def main(args):
    if args.train == 'train':
        dataset = Dataset(directory)
        dataloader = dataset.trainloader()
        optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-4)
        model = ConvLSTM(optimizer=optimizer)
        train(dataloader, model, args.epochs, args.steps_per_epoch)
        utils.save_model(model, save_path)

    else:
        dataset = Dataset(directory)
        dataloader = dataset.testloader()
        model = utils.load_model(save_path)
        test()

if __name__ == '__main__':
    main(args)
