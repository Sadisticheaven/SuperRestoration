import numpy as np
import torch
from torch.backends import cudnn
import utils
from PIL import Image
from imresize import imresize
import os
import niqe
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'ESRGAN'
    rootdir = '../datasets/'
    gnd_data = 'BSDS100/'
    test_data = f'SRGAN_official/{gnd_data}'
    gnd_dir = rootdir + gnd_data
    test_dir = rootdir + test_data
    scale = 4
    padding = scale


    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    imglist = os.listdir(gnd_dir)
    testlist = os.listdir(test_dir)
    Avg_psnr = utils.AverageMeter()
    Avg_niqe = utils.AverageMeter()
    for idx, imgName in enumerate(imglist):
        image = utils.loadIMG_crop(gnd_dir + imgName, scale)
        SR = utils.loadIMG_crop(test_dir + testlist[idx], scale)
        img_mode = image.mode
        if img_mode == 'L':
            gray_img = np.array(image)
        hr_image = np.array(image.convert('RGB')).astype(np.float32)
        hr_image = hr_image[padding: -padding, padding: -padding, ...]

        SR = np.array(SR).astype(np.float32)
        SR = SR[padding: -padding, padding: -padding, ...]
        if img_mode != 'L':
            SR_y = utils.rgb2ycbcr(SR).astype(np.float32)[..., 0] / 255.
            hr_y = utils.rgb2ycbcr(hr_image).astype(np.float32)[..., 0]/255.
        else:
            gray_img = gray_img.astype(np.float32)[padding: -padding, padding: -padding, ...]
            hr_y = gray_img / 255.
            SR = Image.fromarray(SR.astype(np.uint8)).convert('L')
            SR_y = np.array(SR).astype(np.float32) / 255.
        psnr = utils.calc_psnr(hr_y, SR_y)
        NIQE = niqe.calculate_niqe(SR_y)
        Avg_psnr.update(psnr, 1)
        Avg_niqe.update(NIQE, 1)
        print(f'{imgName}, ' + 'PSNR: {:.2f} , NIQE: {:.4f}'.format(psnr.item(), NIQE))
    print('Average_PSNR: {:.2f}, Average_NIQE: {:.4f}'.format(Avg_psnr.avg, Avg_niqe.avg))