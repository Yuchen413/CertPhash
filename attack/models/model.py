import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        orig_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + orig_x
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale1 = nn.Parameter(torch.randn(6, 6))
        self.bias1 = nn.Parameter(torch.randn(4, 6, 6))
        self.conv1 = nn.Conv2d(4, 64, 3, padding="same")
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 5, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, 9, padding="same")

    def forward(self, x):
        x = x.type(torch.float32)
        x = (x - 127.5) / 127.5
        x = x.reshape(-1, 6, 6, 4)
        x = x.permute(0, 3, 1, 2)
        x = torch.mul(x, self.scale1) + self.bias1
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = x * 150
        x = x + 127.5
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)

        return out

class ImageHashingModel(nn.Module):
    def __init__(self, hash_size=144):
        super(ImageHashingModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.layer1 = Bottleneck(64, 64, 256, stride=1)
        self.layer2 = Bottleneck(256, 128, 512, stride=2)
        self.layer3 = Bottleneck(512, 256, 1024, stride=2)
        self.layer4 = Bottleneck(1024, 512, 2048, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, hash_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
