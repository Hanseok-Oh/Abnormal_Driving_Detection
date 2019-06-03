# Pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

# auto encoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # input : 3 * 240 * 320
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), # 유지
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 유지
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 4, stride=2, padding=1), # 축소
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # 유지
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, 4, stride=2, padding=1), # 축소
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1), # 확장
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # 유지
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, stride=1, padding=1), # 유지
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), # 확장
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# auto_encoder = AutoEncoder()
# print(summary(auto_encoder, (3, 240,320)))
