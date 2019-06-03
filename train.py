# Modules
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--array_path", help='array 경로', type=str, nargs='?', default='C:/Users/Yoon/Desktop/프로젝트/이상운전/data/array/dataset_1.npy')

args = parser.parse_args()
array_path = args.array_path
array_data = np.load(array_path)

# custom dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = array_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        return x

dataset = CustomDataset()

# data loader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True)

for idx, data in enumerate(dataloader):
    print(len(data))
