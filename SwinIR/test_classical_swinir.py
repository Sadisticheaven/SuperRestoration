import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm

import utils
from swinir_model import SwinIR
from PIL import Image
from imresize import imresize
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    scale = 4
    model_name = 'SwinIR-Classical'
    weight_file = '../weight_file/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
    dataset = 'degraded_srflow'
    root_dir = f'../datasets/{dataset}/'
    out_root_dir = f'./test_res/{model_name}_{dataset}/'
    hr_dir = '/data0/jli/datasets/PIPAL/'

    lr_dirs = os.listdir(root_dir)
    for dir in lr_dirs:
        utils.mkdirs(out_root_dir + dir)

    model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['params'])
    model.eval()


    for dir in lr_dirs:
        outputs_dir = out_root_dir + dir + '/'
        lr_dir = root_dir + dir + '/'
        lr_lists = os.listdir(lr_dir)
        with tqdm(total=len(lr_lists)) as t:
            t.set_description(f"Processing: {dir}")
            for imgName in lr_lists:
                image = utils.loadIMG_crop(lr_dir + imgName, scale)
                if image.mode != 'L':
                    image = image.convert('RGB')
                lr_image = np.array(image)

                lr = lr_image.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
                lr /= 255.
                lr = torch.from_numpy(lr).to(device).unsqueeze(0)

                with torch.no_grad():
                    SR = model(lr)
                SR = SR.mul(255.0).cpu().numpy().squeeze(0)
                SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])
                # GPU tensor -> CPU tensor -> numpy
                SR = np.array(SR).astype(np.uint8)
                SR = Image.fromarray(SR)  # hw -> wh
                SR.save(outputs_dir + imgName)
                t.update(1)

    sr_dirs = os.listdir(out_root_dir)
    for dir in sr_dirs:
        sr_dir = out_root_dir + dir + '/'
        sr_lists = os.listdir(sr_dir)
        with tqdm(total=len(sr_lists)) as t:
            t.set_description(f"Processing: {dir}")
            for imgName in sr_lists:
                image = utils.loadIMG_crop(sr_dir + imgName, scale)
                if image.mode != 'L':
                    image = image.convert('RGB')
                lr_image = np.array(image)


