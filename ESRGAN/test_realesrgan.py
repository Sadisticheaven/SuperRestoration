import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm
import utils
from realesrgan_model import G
from PIL import Image
import os
import torch.nn.functional as F
from imresize import imresize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'RealESRGAN'
    scale = 2
    itplt = 'bilinear'
    # itplt = 'bicubic'
    # dataset = f'degraded5_{itplt}_medium'
    # dataset = f'degraded6_{itplt}_slight'
    # dataset = f'degraded4_{itplt}_heavy'
    # dataset = f'degraded1_bilinear_heavy'
    # dataset = f'degraded1_offset_bilinear_heavy'
    dataset = f'PIPAL_offset_test'

    weight_file = f'../weight_file/RealESRGAN_x{scale}plus.pth'
    root_dir = f'../datasets/{dataset}/'
    out_root_dir_ImgName = f'./test_res/{model_name}_{dataset}/sort_with_ImgName/'
    out_root_dir_degradation = f'./test_res/{model_name}_{dataset}/sort_with_degradation/'
    hr_dir = '../datasets/PIPAL/'

    lr_dirs = os.listdir(root_dir)
    # Sort with ImgName
    out1_dirs = os.listdir(hr_dir)
    for dir in out1_dirs:
        dir = dir.split('.')[0] + '/'
        utils.mkdirs(out_root_dir_ImgName + dir)
    # Sort with degradation
    for dir in lr_dirs:
        utils.mkdirs(out_root_dir_degradation + dir)

    offset = 1

    if not os.path.exists(weight_file):
        print(f'Weight file not exist!\n{weight_file}\n')
        raise "Error"

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = G(scale=scale).to(device)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['params_ema'])
    model.eval()
    total_dir = len(lr_dirs)
    for idx, dir in enumerate(lr_dirs):
        out_by_deg = out_root_dir_degradation + dir + '/'
        lr_dir = root_dir + dir + '/'
        lr_lists = os.listdir(lr_dir)
        with tqdm(total=len(lr_lists)) as t:
            t.set_description(f"Processing {idx}/{total_dir}: {dir}")
            for imgName in lr_lists:
                out_by_name = out_root_dir_ImgName + imgName.split('.')[0] + '/'
                image = utils.loadIMG_crop(lr_dir + imgName, scale)
                # image = utils.ImgOffSet(image, offset, offset)
                if image.mode == 'L':
                    image = image.convert('RGB')
                lr_image = np.array(image)
                lr_image = imresize(lr_image,  1. / scale, 'bilinear')
                lr = lr_image.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
                lr /= 255.
                lr = torch.from_numpy(lr).to(device).unsqueeze(0)
                lr = F.pad(lr, pad=[1, 1, 1, 1], mode='constant')
                with torch.no_grad():
                    SR = model(lr)
                SR = SR.mul(255.0).cpu().numpy().squeeze(0)
                SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])
                # GPU tensor -> CPU tensor -> numpy
                SR = np.array(SR).astype(np.uint8)
                SR = Image.fromarray(SR)  # hw -> wh
                # SR.save(out_by_name + dir + '.bmp')
                SR.save(out_by_deg + imgName)
                t.update(1)