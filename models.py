import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class CNN_A(nn.Module):
    def __init__(self, output_features: int = 7):
        super(CNN_A, self).__init__()
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(1),
            nn.Linear(512, output_features)
        )

    def forward(self, x):
        return self.conv_stack(x)