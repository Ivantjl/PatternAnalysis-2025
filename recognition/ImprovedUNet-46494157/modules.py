import torch
import torch.nn as nn

class ContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pdrop=0.3):
        super(ContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout2d(p=pdrop)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UpsamplingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        return self.conv(self.upsample(x))


class LocalisationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(LocalisationBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x