import torch
import torch.nn as nn
import torchvision

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

# class coco_photodna(nn.Module):
#     def __init__(self, hash_size=144):
#         super(coco_photodna, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
#
#         self.layer1 = Bottleneck(64, 64, 256, stride=1)
#         self.layer2 = Bottleneck(256, 128, 512, stride=2)
#         self.layer3 = Bottleneck(512, 256, 1024, stride=2)
#         self.layer4 = Bottleneck(1024, 512, 2048, stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(2048, hash_size)
#
#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x

class coco_photodna(torch.nn.Module):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, hash_size=144):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        n_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(n_ftrs, hash_size)
    def forward(self, x):
        logits = self.backbone(x)
        # print(output)
        return logits