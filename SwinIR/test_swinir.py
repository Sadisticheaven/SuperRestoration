import numpy as np
import torch
from torch.backends import cudnn
import utils
from swinir_model import SwinIR
from PIL import Image
from imresize import imresize
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    is_large = False
    if not is_large:
        model_name = 'SwinIR-M'
        weight_file = './weight_file/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
    else:
        model_name = 'SwinIR-L'
        weight_file = './weight_file/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'

    dataset = 'degraded9'
    root_dir = f'/data0/jli/datasets/{dataset}/'
    out_root_dir = f'./test_res/{model_name}_{dataset}/'
    hr_dir = '/data0/jli/datasets/PIPAL/'

    lr_dirs = os.listdir(root_dir)

    out_dirs = os.listdir(hr_dir)
    for dir in out_dirs:
        dir = dir.split('.')[0] + '/'
        utils.mkdirs(out_root_dir+dir)

    scale = 4
    padding = scale
    offset = 0

    if not os.path.exists(weight_file):
        print(f'Weight file not exist!\n{weight_file}\n')
        raise "Error"

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not is_large:
        model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv').to(device)
    else:
        model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv').to(device)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['params_ema'])
    model.eval()

    for dir in lr_dirs:
        # outputs_dir = out_root_dir + dir + '/'
        lr_dir = root_dir + dir + '/'
        lr_lists = os.listdir(lr_dir)
        Avg_psnr = utils.AverageMeter()
        Avg_niqe = utils.AverageMeter()
        for imgName in lr_lists:
            outputs_dir = out_root_dir + imgName.split('.')[0] + '/'
            image = utils.loadIMG_crop(lr_dir + imgName, scale)
            image = utils.ImgOffSet(image, offset, offset)
            hr_image = utils.loadIMG_crop(hr_dir + imgName, scale)
            hr_image = utils.ImgOffSet(hr_image, offset*scale, offset*scale)
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
            # NIQE = niqe.calculate_niqe(SR_y)
            Avg_psnr.update(psnr, 1)
            # Avg_niqe.update(NIQE, 1)
            print(f'{imgName}, ' + 'PSNR: {:.2f}'.format(psnr.item()))
            # print(f'{imgName}, ' + 'PSNR: {:.2f} , NIQE: {:.4f}'.format(psnr.item(), NIQE))
            # GPU tensor -> CPU tensor -> numpy
            output = np.array(SR).astype(np.uint8)
            output = Image.fromarray(output)  # hw -> wh
            # output.save(outputs_dir + imgName)
            output.save(outputs_dir + dir + '.bmp')
            # output.save(outputs_dir + imgName.replace('.', f'_{model_name}_x{scale}.'))
        print('Average_PSNR: {:.2f}, Average_NIQE: {:.4f}'.format(Avg_psnr.avg, Avg_niqe.avg))



