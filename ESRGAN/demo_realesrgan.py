import csv

import numpy as np
import torch
from torch.backends import cudnn
import niqe
import utils
from model import G, G2
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'RealESRGAN'
    weight_file = '../weight_file/RealESRGAN_x4plus.pth'
    img_dir = '../datasets/Real-ESRGAN_input/'
    outputs_dir = './test_res/realESRGAN_x4/'
    utils.mkdirs(outputs_dir)
    scale = 4
    padding = scale

    if not os.path.exists(weight_file):
        print(f'Weight file not exist!\n{weight_file}\n')
        raise "Error"

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = G2().to(device)
    checkpoint = torch.load(weight_file)
    if model_name == 'ESRGAN':
        model.load_state_dict(checkpoint['gen'])
    else:
        model.load_state_dict(checkpoint['params_ema'])
    model.eval()
    imglist = os.listdir(img_dir)

    for imgName in imglist:
        image = utils.loadIMG_crop(img_dir + imgName, scale)
        img_mode = image.mode
        if img_mode == 'L':
            gray_img = np.array(image)
        image = image.convert('RGB')
        lr_image = np.array(image)
        lr = lr_image.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
        lr /= 255.
        lr = torch.from_numpy(lr).to(device).unsqueeze(0)

        with torch.no_grad():
            SR = model(lr)
        SR = SR.mul(255.0).cpu().numpy().squeeze(0)
        SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])
        if img_mode != 'L':
            SR_y = utils.rgb2ycbcr(SR).astype(np.float32)[..., 0] / 255.
        else:
            SR = Image.fromarray(SR.astype(np.uint8)).convert('L')
            SR_y = np.array(SR).astype(np.float32) / 255.
        # GPU tensor -> CPU tensor -> numpy
        output = np.array(SR).astype(np.uint8)
        output = Image.fromarray(output)  # hw -> wh
        output.save(outputs_dir + imgName)




