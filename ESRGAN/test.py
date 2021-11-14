import numpy as np
import torch
from torch.backends import cudnn
import utils
from model import G
from PIL import Image
from imresize import imresize
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'ESRGAN'
    test_data = 'Set5'
    # config = {'weight_file': './',
    config = {'weight_file': './weight_file/ESRGAN_pre_x4_DFO_lr=2e-4_batch=16_out=128/',
    # config = {'weight_file': './weight_file/SRResNet_x4_MSRA_DIV2Kaug_lr=e-4_batch=16_out=96/',
              'img_dir': f'../datasets/{test_data}/',
              'outputs_dir': f'./test_res/test_{model_name}_{test_data}/',
              'scale': 4,
              'visual_filter': False
              }
    outputs_dir = config['outputs_dir']
    scale = config['scale']
    padding = scale
    # weight_file = config['weight_file'] + f'best.pth'
    # weight_file = config['weight_file'] + f'epoch_200.pth'
    weight_file = config['weight_file'] + f'x{scale}/latest.pth'
    img_dir = config['img_dir']
    outputs_dir = outputs_dir + f'x{scale}/'
    utils.mkdirs(outputs_dir)
    if not os.path.exists(weight_file):
        print(f'Weight file not exist!\n{weight_file}\n')
        raise "Error"
    if not os.path.exists(img_dir):
        raise "Image file not exist!\n"

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = G().to(device)
    checkpoint = torch.load(weight_file)
    # if model_name == 'ESRGAN':
    #     model.load_state_dict(checkpoint['gen'])
    # else:
    model.load_state_dict(checkpoint['model'])

    if config['visual_filter']:
        ax = utils.viz_layer(model.extract_layer[0].weight.cpu(), 56)
        ax = utils.viz_layer(model.deconv_layer.weight.cpu(), 56)
    model.eval()
    imglist = os.listdir(img_dir)
    Avg_psnr = utils.AverageMeter()
    for imgName in imglist:
        image = utils.loadIMG_crop(img_dir + imgName, scale)
        img_mode = image.mode
        if img_mode == 'L':
            gray_img = np.array(image)
        image = image.convert('RGB')
        hr_image = np.array(image)

        lr_image = imresize(hr_image, 1. / scale, 'bicubic')
        bic_image = imresize(lr_image, scale, 'bicubic')[padding: -padding, padding: -padding, ...]
        bic_pil = Image.fromarray(bic_image.astype(np.uint8))
        bic_pil.save(outputs_dir + imgName.replace('.', f'_bicubic_x{scale}.'))

        lr = lr_image.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
        lr /= 255.
        lr = torch.from_numpy(lr).to(device).unsqueeze(0)
        hr_image = hr_image[padding: -padding, padding: -padding, ...]

        with torch.no_grad():
            SR = model(lr)
        SR = SR[..., padding: -padding, padding: -padding]
        SR = SR.mul(255.0).cpu().numpy().squeeze(0)
        SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])
        if img_mode != 'L':
            SR_y = utils.rgb2ycbcr(SR).astype(np.float32)[..., 0] / 255.
            hr_y = utils.rgb2ycbcr(hr_image).astype(np.float32)[..., 0]/255.
        else:
            hr_y = gray_img.astype(np.float32)[padding: -padding, padding: -padding, ...] / 255.
            SR = Image.fromarray(SR.astype(np.uint8)).convert('L')
            SR_y = np.array(SR).astype(np.float32) / 255.
        psnr = utils.calc_psnr(hr_y, SR_y)
        Avg_psnr.update(psnr, 1)
        print(f'{imgName}, ' + 'PSNR: {:.2f}'.format(psnr.item()))
        # GPU tensor -> CPU tensor -> numpy
        output = np.array(SR).astype(np.uint8)
        output = Image.fromarray(output) # hw -> wh
        output.save(outputs_dir + imgName.replace('.', f'_{model_name}_x{scale}.'))
    print('Average_PSNR: {:.2f}'.format(Avg_psnr.avg))