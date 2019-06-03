import torch
from torch.utils.data import Dataset

# custom dataset
class CustomDataset(Dataset):
    def __init__(self, x_data):
        self.x_data = x_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        return x
