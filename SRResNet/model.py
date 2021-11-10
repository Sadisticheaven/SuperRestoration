import math
from torch import nn
import torch
padding_mode = 'replicate'


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode=padding_mode, bias=False)
        self.prelu = nn.PReLU()
        self.resblock = self.make_layer(_Residual_Block, 16)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4, padding_mode=padding_mode, bias=False)

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
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # in:48*48
            nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # in:24*24
            nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # in:12*12
            nn.Conv2d(256, 512, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # in:6*6
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def init_weight(self):
        for L in self.modules():
            if isinstance(L, nn.Conv2d):
                n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
                L.weight.data.normal_(0, math.sqrt(2. / n))
                if L.bias is not None:
                    L.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.dense(x)
        return torch.sigmoid(x)


def test():
    low_resolution = 24  # 96x96 -> 24x24
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = G()
        gen_out = gen(x)
        disc = D()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)


if __name__ == "__main__":
    test()






