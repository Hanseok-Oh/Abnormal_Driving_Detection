# Modules
import torch
import numpy as np
from datetime import datetime

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchsummary import summary

# model
from modules.data.load import get_dataset
from modules.models import autoencoder

# argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", help='video path', type=str)
#parser.add_argument("--weight_path", help='weight path', type=str)

args = parser.parse_args()
video_path = args.video_path
#weight_path = args.weight_path

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(device))


# dataset
dataset = get_dataset(video_path, offset=10)

batch = 2
dataloader = DataLoader(
    dataset,
    batch_size=batch,
    shuffle=True)


# model
model = autoencoder.AutoEncoder()
model.to(device)
print(summary(model, (3, 240,320)))

# train
criterion = nn.MSELoss().to(device) # loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3) # adam optimizer
epoch = 1

print("학습 시작")
start_time = datetime.now()
for ep in range(epoch):
    ep_loss = 0
    for idx, data in enumerate(dataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()
        ep_loss += loss.item()

    ep_time = datetime.now() - start_time
    ep_loss = loss.item() / batch
    print("epoch:{} / loss:{} / time:{}".format(ep, ep_loss, ep_time))
print("학습 완료")

# 모델 저장
#torch.save(model.state_dict(), weight_path)
