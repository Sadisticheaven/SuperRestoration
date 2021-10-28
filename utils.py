# PyTorch
import copy

import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


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