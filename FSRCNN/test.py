import numpy as np
import torch
from torch.backends import cudnn
import utils
from model import FSRCNN, N2_10_4
from PIL import Image
from imresize import imresize
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    # config = {'weight_file': './weight_file/FSRCNN_x3_XavierTanh_T91_lr=e-1_batch=64_input=11/',
    # config = {'weight_file': './weight_file/N2-10-4_x3_MSRA_G191res_lr=5e-1_batch=512_out=19/',
    # config = {'weight_file': './weight_file/N2-10-4_x3_MSRA_T91resMod_lr=1_batch=512_out=19/',
    # config = {'weight_file': './weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=1_batch=128_Huber=9e-1/',
    # config = {'weight_file': './weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_out=19/',
    # config = {'weight_file': './weight_file/Test5/N2-10-4_x3_MSRA_G191res_lr=e-2_batch=128_CLoss=e-3/',
    # config = {'weight_file': './weight_file/Test7/N2-10-4_x4_MSRA_G191res_lr=e-2_batch=128_CLoss=e-3/',
    config = {'weight_file': './weight_file/Test1/FSRCNN_x3_MSRA_T91_lr=e-1_batch=128_input=11/',
    #           'img_dir': '../datasets/BSDS200/',
              'img_dir': '../datasets/Set14/',
              # 'outputs_dir': './test_res/test_11-27_BSDS200/',
              # 'outputs_dir': './test_res/test_N2-10-4-G191_Set14/',
              'outputs_dir': './test_res/test_FSRCNN_X3_Set14/',
              # 'outputs_dir': './test_res/test_XavierTanh_Set14/',
              # 'outputs_dir': './test_res/test_191res_Set14/',
              'in_size': 11,
              'out_size': 19,
              'scale': 3,
              'd': 10,
              'm': 4,
              'residual': False,
              'visual_filter': False
              }
    method = 'bicubic'
    outputs_dir = config['outputs_dir']
    scale = config['scale']
    in_size = config['in_size']
    out_size = config['out_size']

    padding = abs(in_size * scale - out_size)//2
    # padding = scale
    weight_file = config['weight_file'] + f'best.pth'
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

    # model = N2_10_4(scale, in_size, out_size, config['d'], config['m']).to(device)
    model = FSRCNN(scale, in_size, out_size).to(device)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['model'])

    if config['visual_filter']:
        ax = utils.viz_layer(model.extract_layer[0].weight.cpu(), 56)
        ax = utils.viz_layer(model.deconv_layer.weight.cpu(), 56)
    model.eval()
    imglist = os.listdir(img_dir)
    Avg_psnr = utils.AverageMeter()
    for imgName in imglist:
        image = utils.loadIMG_crop(img_dir + imgName, scale)
        if image.mode != 'L':  # gray image don't need to convert
            image = image.convert('RGB')
        hr_image = np.array(image)

        lr_image = imresize(hr_image, 1. / scale, method)
        bic_image = imresize(lr_image, scale, method)
        bic_image = bic_image[padding: -padding, padding: -padding, ...]
        bic_pil = Image.fromarray(bic_image.astype(np.uint8))
        bic_pil.save(outputs_dir + imgName.replace('.', f'_{method}_x{scale}.'))

        hr_image = hr_image[padding: -padding, padding: -padding, ...]

        lr_y, _ = utils.preprocess(lr_image, device, image.mode)
        hr_y, _ = utils.preprocess(hr_image, device, image.mode)
        bic_y, ycbcr = utils.preprocess(bic_image, device, image.mode)
        with torch.no_grad():
            SR = model(lr_y)
        if scale != 3:
            SR = SR[..., 1:, 1:]
        if config['residual']:
            SR = SR + bic_y
        SR = SR.clamp(0.0, 1.0)
        psnr = utils.calc_psnr(hr_y, SR)
        Avg_psnr.update(psnr, 1)
        print(f'{imgName}, ' + 'PSNR: {:.2f}'.format(psnr.item()))
        # GPU tensor -> CPU tensor -> numpy
        SR = SR.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        if image.mode == 'L':
            output = np.clip(SR, 0.0, 255.0).astype(np.uint8)  # chw -> hwc
        else:
            ycbcr = ycbcr.mul(255.0).cpu().numpy().squeeze(0).transpose([1, 2, 0])
            output = np.array([SR, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0]) # chw -> hwc
            output = np.clip(utils.ycbcr2rgb(output), 0.0, 255.0).astype(np.uint8)
        output = Image.fromarray(output) # hw -> wh
        output.save(outputs_dir + imgName.replace('.', f'_FSRCNN_x{scale}.'))
    print('Average_PSNR: {:.2f}'.format(Avg_psnr.avg))