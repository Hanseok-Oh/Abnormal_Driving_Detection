import argparse
import utils
from utils.dataloader import DataLoader

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--directory', help='데이터 디렉토리', type=str)
parser.add_argument('--batch_size', help='batch size', type=int, default=32)
parser.add_argument('--batch_per_video', help='batch per video', type=int, default=4)
args = parser.parse_args()


def train():
    return

def test():
    return

def main(args):
    loader = DataLoader(args.directory).random_batches()
    x, y = next(loader)
    print(x.shape)


if __name__ == '__main__':
    main(args)
