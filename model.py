import torch.nn as nn
import torch
from torchvision import models

class ResNetIrisTumor(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(ResNetIrisTumor, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
