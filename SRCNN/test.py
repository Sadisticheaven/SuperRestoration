import numpy as np
import torch
from torch.backends import cudnn
import utils
from models import SRCNN
from PIL import Image
import os
from imresize import imresize
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':

    # config = {'weight_file': './weight_file/SRCNN_x3_data=matlab_lr=1e-2_batch=128/',
    config = {'weight_file': './weight_file/SRCNN_x3_lr=1e-02_batch=128/',
              'img_dir': '../datasets/Set14/',
              'outputs_dir': './test_res/test_915_lr=1e-2_x3_Set14/',
              'scale': 3,
              'visual_filter': False
              }

    outputs_dir = config['outputs_dir']
    scale = config['scale']
    # weight_file = config['weight_file'] + f'best.pth'
    # weight_file = config['weight_file'] + f'SRCNNx3_data=276864_lr=1e-2.pth'
    weight_file = config['weight_file'] + f'x{scale}/best.pth'
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

    # state_dict = model.state_dict()
    # for n, p in torch.load(weight_file, map_location=lambda storage, loc: storage).items():
    #     if n in state_dict.keys():
    #         state_dict[n].copy_(p)
    #     else:
    #         raise KeyError(n)
    if config['visual_filter']:
        for L in model.conv1:
            if isinstance(L, nn.Conv2d):
                ax = utils.viz_layer(L.weight.cpu(), 64)
        # ax = utils.viz_layer(model.conv1.weight.cpu(), 64)
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
        gnd_image = np.array(image)

        lr_image = imresize(gnd_image, 1. / scale, 'bicubic')
        bic_image = imresize(lr_image, scale, 'bicubic')
        shave = scale
        bic_pil = Image.fromarray(bic_image.astype(np.uint8)[shave: -shave, shave: -shave, ...])
        bic_pil.save(outputs_dir + imgName.replace('.', f'_bicubic_x{scale}.'))

        bic_y, bic_ycbcr = utils.preprocess(bic_image, device, image.mode)
        gnd_y, _ = utils.preprocess(gnd_image, device, image.mode)
        with torch.no_grad():
            preds = model(bic_y).clamp(0.0, 1.0)

        gnd_y = gnd_y[..., shave: -shave, shave: -shave]
        preds = preds[..., shave: -shave, shave: -shave]
        bic_ycbcr = bic_ycbcr[shave: -shave, shave: -shave, ...]
        bic_y = bic_y[..., shave: -shave, shave: -shave]

        psnr = utils.calc_psnr(gnd_y, preds)
        psnr2 = utils.calc_psnr(gnd_y, bic_y)
        Avg_psnr.update(psnr, 1)
        # Avg_psnr.update(psnr2, 1)
        print(f'{imgName}, ' + 'PSNR: {:.2f}'.format(psnr.item()))
        print(f'{imgName}, ' + 'PSNR_bicubic: {:.2f}'.format(psnr2.item()))
        # GPU tensor -> CPU tensor -> numpy
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        if image.mode == 'L':
            output = np.clip(preds, 0.0, 255.0).astype(np.uint8)  # chw -> hwc
        else:
            output = np.array([preds, bic_ycbcr[..., 1], bic_ycbcr[..., 2]]).transpose([1, 2, 0])  # chw -> hwc
            output = np.clip(utils.ycbcr2rgb(output), 0.0, 255.0).astype(np.uint8)
        output = Image.fromarray(output)  # hw -> wh

        output.save(outputs_dir + imgName.replace('.', f'_SRCNNx{scale}.'))
    print('Average_PSNR: {:.2f}'.format(Avg_psnr.avg))