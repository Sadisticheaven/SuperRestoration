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

    config = {'weight_file': './',
              'img_dir': '../datasets/Set5/',
              'outputs_dir': './test_res/test_SRResNet_Set5/',
              'in_size': 24,
              'out_size': 96,
              'scale': 4,
              'visual_filter': False
              }

    outputs_dir = config['outputs_dir']
    scale = config['scale']
    in_size = config['in_size']
    out_size = config['out_size']
    padding = scale
    # weight_file = config['weight_file'] + f'best.pth'
    # weight_file = config['weight_file'] + f'FSRCNNx3_lr=e-2_91img.pth'
    weight_file = config['weight_file'] + f'latest.pth'
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

    model = G().to(device)
    # model = FSRCNN(scale, in_size, out_size).to(device)
    checkpoint = torch.load(weight_file)
    if len(checkpoint) < 6:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    if config['visual_filter']:
        ax = utils.viz_layer(model.extract_layer[0].weight.cpu(), 56)
        ax = utils.viz_layer(model.deconv_layer.weight.cpu(), 56)
    model.eval()
    imglist = os.listdir(img_dir)
    Avg_psnr = utils.AverageMeter()
    for imgName in imglist:
        img_file = img_dir + imgName
        image = Image.open(img_file)  # (width, height)
        lr_wid = image.width // scale
        lr_hei = image.height // scale
        image = image.crop((0, 0, lr_wid * scale, lr_hei * scale))
        if image.mode != 'L':  # gray image don't need to convert
            image = image.convert('RGB')
        hr_image = np.array(image)

        lr_image = imresize(hr_image, 1. / scale, 'bicubic')
        bic_image = imresize(lr_image, scale, 'bicubic')
        bic_image = bic_image[padding: -padding, padding: -padding, ...]
        bic_pil = Image.fromarray(bic_image.astype(np.uint8))
        bic_pil.save(outputs_dir + imgName.replace('.', f'_bicubic_x{scale}.'))

        hr_image = hr_image[padding: -padding, padding: -padding, ...]

        # _, lr = utils.preprocess(lr_image, device, image.mode)
        lr = np.array(lr_image).astype(np.float32).transpose([2, 0, 1])  # (width, height, channel) -> (height, width, channel)
        lr /= 255.
        lr = torch.from_numpy(lr).to(device).unsqueeze(0)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)
        preds = preds[..., padding: -padding, padding: -padding]
        preds = preds.mul(255.0).cpu().numpy().squeeze(0)
        preds = np.clip(preds, 0.0, 255.0).transpose([1, 2, 0])
        preds_y = utils.rgb2ycbcr(preds).astype(np.float32)[..., 0]/255.
        hr_y = utils.rgb2ycbcr(hr_image).astype(np.float32)[..., 0]/255.
        psnr = utils.calc_psnr(hr_y, preds_y)
        # psnr2 = utils.calc_psnr(hr_y, bic_y)
        Avg_psnr.update(psnr, 1)
        print(f'{imgName}, ' + 'PSNR: {:.2f}'.format(psnr.item()))
        # print(f'{imgName}, ' + 'PSNR_bic: {:.2f}'.format(psnr2.item()))
        # GPU tensor -> CPU tensor -> numpy
        output = preds.astype(np.uint8)
        output = Image.fromarray(output) # hw -> wh
        output.save(outputs_dir + imgName.replace('.', f'_FSRCNN_x{scale}.'))
    print('Average_PSNR: {:.2f}'.format(Avg_psnr.avg))