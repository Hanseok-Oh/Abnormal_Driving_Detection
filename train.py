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
from torchsummary import summary

# model
from modules.data import customdataset
from modules.models import autoencoder

# argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--array_path", help='array path', type=str)
parser.add_argument("--weight_path", help='weight path', type=str)

args = parser.parse_args()
array_path = args.array_path
weight_path = args.weight_path

array_data = np.load(array_path)
array_data = np.transpose(array_data, (0, 3, 1, 2))

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(device))

# load data
batch = 2
dataset = customdataset.CustomDataset(x_data=array_data)
dataloader = DataLoader(
    dataset,
    batch_size=batch,
    shuffle=True)

# model
model = autoencoder.AutoEncoder()
model.to(device)
#print(summary(model, (3, 240,320)))

# train
criterion = nn.MSELoss().to(device) # loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3) # adam optimizer
epoch = 2

print("학습 시작")
start_time = datetime.now()
for ep in range(epoch):
    ep_loss = 0
    for idx, data in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        loss = criterion(outputs, data)
        loss.backward()

        optimizer.step()
        ep_loss += loss.item()

    ep_time = datetime.now() - start_time
    ep_loss = loss.item() / batch
    print("epoch:{} / loss:{} / time:{}".format(ep, ep_loss, ep_time))
print("학습 완료")

# 모델 저장
torch.save(model.state_dict(), weight_path)
