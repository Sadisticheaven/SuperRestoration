import csv

import numpy as np
import torch
from torch.backends import cudnn

import niqe
import utils
from bsrgan_model import RRDBNet
from PIL import Image
from imresize import imresize
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'BSRGAN'
    weight_file = './weight_file/BSRGAN.pth'
    root_dir = '/data0/jli/datasets/degraded_17/'
    out_root_dir = f'./test_res/{model_name}_degraded17/'
    hr_dir = '/data0/jli/datasets/PIPAL/'

    csv_file = f'./test_res/{model_name}_x4_degraded17.csv'
    csv_file = open(csv_file, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(('name', 'psnr', 'niqe', 'ssim'))

    lr_dirs = os.listdir(root_dir)
    for dir in lr_dirs:
        utils.mkdirs(out_root_dir + dir)

    scale = 4
    padding = scale


    if not os.path.exists(weight_file):
        print(f'Weight file not exist!\n{weight_file}\n')
        raise "Error"

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RRDBNet().to(device)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint)
    model.eval()

    for dir in lr_dirs:
        outputs_dir = out_root_dir + dir + '/'
        lr_dir = root_dir + dir + '/'
        lr_lists = os.listdir(lr_dir)
        Avg_psnr = utils.AverageMeter()
        Avg_niqe = utils.AverageMeter()
        Avg_ssim = utils.AverageMeter()
        for imgName in lr_lists:
            image = utils.loadIMG_crop(lr_dir + imgName, scale)
            hr_image = utils.loadIMG_crop(hr_dir + imgName, scale)
            img_mode = image.mode
            if img_mode == 'L':
                gray_img = np.array(image)
            image = image.convert('RGB')
            lr_image = np.array(image)
            hr_image = np.array(hr_image)

            lr = lr_image.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
            lr /= 255.
            lr = torch.from_numpy(lr).to(device).unsqueeze(0)

            # hr_image = hr_image[padding: -padding, padding: -padding, ...]

            with torch.no_grad():
                SR = model(lr)
            # SR = SR[..., padding: -padding, padding: -padding]
            SR = SR.mul(255.0).cpu().numpy().squeeze(0)
            SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])
            if img_mode != 'L':
                SR_y = utils.rgb2ycbcr(SR).astype(np.float32)[..., 0] / 255.
                hr_y = utils.rgb2ycbcr(hr_image).astype(np.float32)[..., 0] / 255.
            else:
                # gray_img = gray_img.astype(np.float32)[padding: -padding, padding: -padding, ...]
                hr_y = gray_img / 255.
                SR = Image.fromarray(SR.astype(np.uint8)).convert('L')
                SR_y = np.array(SR).astype(np.float32) / 255.
            psnr = utils.calc_psnr(hr_y, SR_y)
            NIQE = niqe.calculate_niqe(SR_y)
            ssim = utils.calculate_ssim(hr_y * 255, SR_y * 255)
            Avg_psnr.update(psnr, 1)
            Avg_niqe.update(NIQE, 1)
            Avg_ssim.update(ssim, 1)
            # print(f'{imgName}, ' + 'PSNR: {:.2f}'.format(psnr.item()))
            print(f'{imgName}, ' + 'PSNR: {:.2f} , NIQE: {:.4f}, ssim: {:.4f}'.format(psnr.item(), NIQE, ssim))
            # GPU tensor -> CPU tensor -> numpy
            output = np.array(SR).astype(np.uint8)
            output = Image.fromarray(output)  # hw -> wh
            output.save(outputs_dir + imgName.replace('.', '_{:.2f}.'.format(psnr.item())))
        print('Average_PSNR: {:.2f}, Average_NIQE: {:.4f}, Average_ssim: {:.4f}'.format(Avg_psnr.avg, Avg_niqe.avg,
                                                                                        Avg_ssim.avg))
        writer.writerow((dir, Avg_psnr.avg, Avg_niqe.avg, Avg_ssim.avg))




