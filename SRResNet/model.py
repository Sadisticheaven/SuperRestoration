import math
from torch import nn
import torch
padding_mode = 'replicate'
class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(num_features=64, affine=True)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(num_features=64, affine=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = identity + x
        return x


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode=padding_mode)
        self.prelu = nn.PReLU()
        self.resblock = self.make_layer(_Residual_Block, 16)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(num_features=64, affine=True)
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4, padding_mode=padding_mode)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def init_weight(self):
        for L in self.modules():
            if isinstance(L, nn.Conv2d):
                n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
                L.weight.data.normal_(0, math.sqrt(2. / n))
                if L.bias is not None:
                    L.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        identity = x
        x = self.resblock(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = identity + x
        x = self.upscale4x(x)
        x = self.conv3(x)
        return x


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.features = nn.Sequential(
            # in:96*96
            nn.Conv2d(3, 64, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # in:96*96
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # in:48*48
            nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # in:24*24
            nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # in:12*12
            nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # in:6*6
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.Lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.features(x)
        x = self.fc1(x)
        x = self.Lrelu(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x








