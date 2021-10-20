import math
from torch import nn
import torch

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, in_size, out_size, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.extract_layer = nn.Sequential(nn.Conv2d(num_channels, d, kernel_size=5, padding=2, padding_mode='replicate'),
                                           nn.PReLU())

        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU()]
        for i in range(m):
            self.mid_part.extend([nn.ReplicationPad2d(1),
                                  nn.Conv2d(s, s, kernel_size=3),
                                  nn.PReLU()])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU()])
        self.mid_part = nn.Sequential(*self.mid_part)

        # 11->out
        self.deconv_layer = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor,
                                               padding=(9 + (in_size-1)*scale_factor - out_size)//2)

    def init_weights(self, method='MSRA'):
        if method == 'MSRA':
            self._init_weights_MSRA()
        else:
            self._init_weights_Xavier()

    def _init_weights_MSRA(self):
        for L in self.extract_layer:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels * L.weight.data[0][0].numel())))
                L.bias.data.zero_()
        for L in self.mid_part:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels * L.weight.data[0][0].numel())))
                L.bias.data.zero_()
        self.deconv_layer.weight.data.normal_(mean=0.0, std=0.001)
        self.deconv_layer.bias.data.zero_()

    def _init_weights_Xavier(self):
        for L in self.extract_layer:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels + L.in_channels)))
                L.bias.data.zero_()
        for L in self.mid_part:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels + L.in_channels)))
                L.bias.data.zero_()
        self.deconv_layer.weight.data.normal_(mean=0.0, std=0.001)
        self.deconv_layer.bias.data.zero_()

    def forward(self, x):
        x = self.extract_layer(x)
        x = self.mid_part(x)
        x = self.deconv_layer(x)
        return x


class N2_10_4(nn.Module):
    def __init__(self, scale_factor, in_size, out_size, num_channels=1, d=10, m=4):
        super(N2_10_4, self).__init__()
        self.extract_layer = nn.Sequential(nn.Conv2d(num_channels, d, kernel_size=3, padding=1, padding_mode='replicate'),
                                           nn.PReLU())
        self.mid_part = []
        for i in range(m):
            self.mid_part.extend([nn.Conv2d(d, d, kernel_size=3, padding=1, padding_mode='replicate'),
                                  nn.PReLU()])
        self.mid_part = nn.Sequential(*self.mid_part)

        # 11->out
        self.deconv_layer = nn.ConvTranspose2d(d, num_channels, kernel_size=7, stride=scale_factor,
                                               padding=(7 + (in_size-1)*scale_factor - out_size)//2)

    def forward(self, x):
        x = self.extract_layer(x)
        x = self.mid_part(x)
        x = self.deconv_layer(x)
        return x

    def init_weights(self, method='MSRA'):
        if method == 'MSRA':
            self._init_weights_MSRA()
        else:
            self._init_weights_Xavier()

    def _init_weights_MSRA(self):
        for L in self.extract_layer:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels * L.weight.data[0][0].numel())))
                L.bias.data.zero_()
        for L in self.mid_part:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels * L.weight.data[0][0].numel())))
                L.bias.data.zero_()
        self.deconv_layer.weight.data.normal_(mean=0.0, std=0.001)
        self.deconv_layer.bias.data.zero_()

    def _init_weights_Xavier(self):
        for L in self.extract_layer:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels + L.in_channels)))
                L.bias.data.zero_()
        for L in self.mid_part:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=math.sqrt(2 / (L.out_channels + L.in_channels)))
                L.bias.data.zero_()
        self.deconv_layer.weight.data.normal_(mean=0.0, std=0.001)
        self.deconv_layer.bias.data.zero_()


class CLoss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self, delta=1e-3):
        super(CLoss, self).__init__()
        self.delta2 = delta * delta

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.delta2)
        loss = torch.mean(error)
        return loss