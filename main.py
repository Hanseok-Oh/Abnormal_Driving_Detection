# Modules
import argparse
import numpy as np
from datetime import datetime

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsummary import summary

# model
from modules.data.load import load_dataset
from modules.models import autoencoder

# 실행 함수
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help='비디오 저장 폴더', type=str)
    parser.add_argument("--weight_path", help='모델 weight 저장할 폴더', type=str, default='temp')
    parser.add_argument("--offset", help='X, Y frame 차이', type=int, default=10)
    parser.add_argument("--batch_size", help='Batch size', type=int, default=32)
    parser.add_argument("--step_size", help='학습시킬 비디오 개수', type=int, default=16)

    args = parser.parse_args()
    train(args)

# GPU 사용 확인
def check_device():
    # gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("현재 사용 Device : {}".format(device))
    return device

def get_dataset_loader(args):
    transform = torchvision.transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset, dataloader = load_dataset(
        video_path = args.video_path,
        batch_size = args.batch_size,
        offset = args.offset,
        transform = transform)
    return dataset, dataloader

def train(args):
    device = check_device()

    generative_model = autoencoder.AutoEncoder()
    generative_model = generative_model.to(device)
    print(summary(generative_model, (3,240,320)))

    criterion = nn.MSELoss().to(device) # loss function
    optimizer = optim.Adam(generative_model.parameters(), lr=1e-3) # adam optimizer

    print('학습 시작')
    start_time = datetime.now()

    for st in range(args.step_size):
        dataset, dataloader = get_dataset_loader(args) # load video frame
        step_loss = 0.0
        step_total = 0.0

        for idx, data in enumerate(dataloader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = generative_model(x)

            loss = criterion(outputs, x)
            loss.backward()
            optimizer.step()

            step_loss += loss.item()
            step_total += x.size(0)

        step_time = datetime.now() - start_time
        step_loss /= step_total

        print("step:{} / loss:{} / time:{}".format(st, step_loss, step_time))
    print("학습 완료")

if __name__ == '__main__':
    main()


# 모델 저장
#torch.save(model.state_dict(), weight_path)
