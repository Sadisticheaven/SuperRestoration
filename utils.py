# PyTorch
import copy

import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loadIMG(img_path,):
    hrIMG = Image.open(img_path)

    lr_wid = hrIMG.width // scale
    lr_hei = hrIMG.height // scale
    hr_wid = lr_wid * scale
    hr_hei = lr_hei * scale

    hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei))

    if hrIMG.mode == 'L':
        hr = np.array(hrIMG)
        hr = hr.astype(np.float32)
    else:
        hrIMG.convert('RGB')
        hr = np.array(hrIMG)
        hr = utils.rgb2ycbcr(hr).astype(np.float32)[..., 0]

def calc_psnr(img1, img2):
    if isinstance(img1, torch.Tensor):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
    else:
        return 10. * np.log10(1. / np.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def preprocess(img, device, image_mode='RGB'):
    if image_mode == 'RGB':
        img = np.array(img).astype(np.float32)  # (width, height, channel) -> (height, width, channel)
        ycbcr = rgb2ycbcr(img).astype(np.float32).transpose([2, 0, 1])
        ycbcr /= 255.
        ycbcr = torch.from_numpy(ycbcr).to(device).unsqueeze(0)  # numpy -> cpu tensor -> GPU tensor
        y = ycbcr[0, 0, ...].unsqueeze(0).unsqueeze(0)  # input Tensor Dimension: batch_size * channel * H * W
        return y, ycbcr
    else:
        y = img.astype(np.float32)  # (width, height, channel) -> (height, width, channel)
        y /= 255.
        y = torch.from_numpy(y).to(device)  # numpy -> cpu tensor -> GPU tensor
        y = y.unsqueeze(0).unsqueeze(0)  # input Tensor Dimension: batch_size * channel * H * W
        return y, y


# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters=4):
    plt.figure(figsize=(4, 3.5))
    min = torch.min(layer).item()
    max = torch.max(layer).item()
    # mean = torch.mean(layer).item()
    # std = torch.std(layer).item()
    # transforms1 = transforms.Normalize(mean=mean, std=std)
    for index, filter in enumerate(layer):
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.subplot(8, 8, index + 1)
        # min = torch.min(filter).item()
        # max = torch.max(filter).item()
        filter = (filter - min)/(max - min)
        # plt.imshow(transforms1(filter.detach())[0, :, :],  cmap='gray')
        plt.imshow(filter[0, :, :].detach(),  cmap='gray')
        plt.axis('off')
    plt.show()


def viz_layer2(layer, n_filters=4):
    plt.figure(figsize=(4, 3.5))
    layer = torch.from_numpy(layer)
    min = torch.min(layer).item()
    max = torch.max(layer).item()
    # mean = torch.mean(layer).item()
    # std = torch.std(layer).item()
    # transforms1 = transforms.Normalize(mean=mean, std=std)
    for index in range(n_filters):
        filter = layer[:, :, index]
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.subplot(8, 8, index + 1)
        # filter = (filter - min)/(max - min)
        # plt.imshow(transforms1(filter.detach())[0, :, :],  cmap='gray')
        plt.imshow(filter[:, :].detach(),  cmap='gray')
        plt.axis('off')
    plt.show()


def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112.0],
                  [ 112.0, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    ycbcr = np.round(ycbcr)
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    rgb = np.round(rgb)
    return rgb.clip(0, 255).reshape(shape)