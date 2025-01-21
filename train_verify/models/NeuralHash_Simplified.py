import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv2dDynamicSamePadding(nn.Conv2d):
    """Custom convolutional layer with dynamic padding."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
       return F.relu(x + 3)

class NeuralHash_Simplified(nn.Module):
    def __init__(self, in_ch=3,in_dim=32):
        super(NeuralHash_Simplified, self).__init__()
        
        # Initial convolution with dynamic padding
        self.conv1 = Conv2dDynamicSamePadding(in_ch, 16, kernel_size=1, stride=1)

        # Bottleneck block 1
        self.bottleneck1 = nn.Sequential(
            Conv2dDynamicSamePadding(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            Conv2dDynamicSamePadding(32, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            CustomActivation()
        )

        # Bottleneck block 2
        self.bottleneck2 = nn.Sequential(
            Conv2dDynamicSamePadding(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            Conv2dDynamicSamePadding(32, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            CustomActivation()
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 10, kernel_size=1, stride=1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.output_layer(x)
        return x

