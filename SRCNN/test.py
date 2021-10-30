import numpy as np
import torch
from torch.backends import cudnn
import utils
from models import SRCNN
from PIL import Image
import os
from imresize import imresize

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    config = {'weight_file': './weight_file/',
              'img_dir': '../datasets/Set5/',
              'outputs_dir': './test_res/test_x4_Set5/',
              'scale': 4,
              'visual_filter': True
              }

    outputs_dir = config['outputs_dir']
    scale = config['scale']
    padding = scale
    weight_file = config['weight_file'] + f'best.pth'
    # weight_file = config['weight_file'] + f'SRCNNx3_data=276864_lr=1e-2.pth'
    # weight_file = config['weight_file'] + f'x{scale}/best.pth'
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

    model = SRCNN(padding=True).to(device)
    checkpoint = torch.load(weight_file)
    if len(checkpoint) < 6:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    if config['visual_filter']:
        ax = utils.viz_layer(model.conv1[0].weight.cpu(), 64)
    model.eval()
    imglist = os.listdir(img_dir)
    Avg_psnr = utils.AverageMeter()
    for imgName in imglist:
        image = utils.loadIMG_crop(img_dir + imgName, scale)
        if image.mode != 'L':  # gray image don't need to convert
            image = image.convert('RGB')
        hr_image = np.array(image)

        lr_image = imresize(hr_image, 1. / scale, 'bicubic')
        bic_image = imresize(lr_image, scale, 'bicubic')
        bic_pil = Image.fromarray(bic_image.astype(np.uint8)[padding: -padding, padding: -padding, ...])
        bic_pil.save(outputs_dir + imgName.replace('.', f'_bicubic_x{scale}.'))

        bic_y, bic_ycbcr = utils.preprocess(bic_image, device, image.mode)
        hr_y, _ = utils.preprocess(hr_image, device, image.mode)
        with torch.no_grad():
            SR = model(bic_y).clamp(0.0, 1.0)
        hr_y = hr_y[..., padding: -padding, padding: -padding]
        SR = SR[..., padding: -padding, padding: -padding]
        bic_ycbcr = bic_ycbcr[..., padding: -padding, padding: -padding]
        bic_y = bic_y[..., padding: -padding, padding: -padding]

        psnr = utils.calc_psnr(hr_y, SR)
        psnr2 = utils.calc_psnr(hr_y, bic_y)
        Avg_psnr.update(psnr, 1)
        # Avg_psnr.update(psnr2, 1)
        print(f'{imgName}, ' + 'PSNR: {:.2f}'.format(psnr.item()))
        print(f'{imgName}, ' + 'PSNR_bicubic: {:.2f}'.format(psnr2.item()))
        # GPU tensor -> CPU tensor -> numpy
        SR = SR.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        if image.mode == 'L':
            output = np.clip(SR, 0.0, 255.0).astype(np.uint8)  # chw -> hwc
        else:
            bic_ycbcr = bic_ycbcr.mul(255.0).cpu().numpy().squeeze(0).transpose([1, 2, 0])
            output = np.array([SR, bic_ycbcr[..., 1], bic_ycbcr[..., 2]]).transpose([1, 2, 0])  # chw -> hwc
            output = np.clip(utils.ycbcr2rgb(output), 0.0, 255.0).astype(np.uint8)
        output = Image.fromarray(output)  # hw -> wh
        output.save(outputs_dir + imgName.replace('.', f'_SRCNNx{scale}.'))
    print('Average_PSNR: {:.2f}'.format(Avg_psnr.avg))