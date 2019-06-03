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
parser.add_argument("--array_path", help='array 경로', type=str, nargs='?', default='C:/Users/Yoon/Desktop/프로젝트/이상운전/data/array/dataset_1.npy')

args = parser.parse_args()
array_path = args.array_path
array_data = np.load(array_path)
array_data = np.transpose(array_data, (0, 3, 1, 2))

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(device))

# load data
dataset = customdataset.CustomDataset(x_data=array_data)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True)

# model
auto_encoder = autoencoder.AutoEncoder()
#print(summary(auto_encoder, (3, 240,320)))

# train
criterion = nn.MSELoss().to(device) # loss function
optimizer = optim.Adam(auto_encoder.parameters(), lr=1e-3) # adam optimizer
epoch = 3

start_time = datetime.now()
for ep in range(epoch):
    for idx, data in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        outputs = auto_encoder(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()

    print(loss.item())
