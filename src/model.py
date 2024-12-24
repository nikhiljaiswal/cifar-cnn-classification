import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10, dropout_value=0.05):
        super(CIFAR10Net, self).__init__()

        # C1 Block
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        # C2 Block (Depthwise Separable Convolution)
        self.c2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=24, stride=1, bias=False),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        # C3 Block (Dilated Convolution)
        self.c3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=2, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # C4 Block
        self.c4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 64, bias=False)
        self.dropout = nn.Dropout(dropout_value)
        self.fc2 = nn.Linear(64, num_classes, bias=False)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 