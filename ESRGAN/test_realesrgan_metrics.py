import csv
import metrics
import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm
import utils
from realesrgan_model import G
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'RealESRGAN'
    itplt = 'bilinear'
    # itplt = 'bicubic'
    dataset = f'degraded4_{itplt}_heavy'
    # dataset = f'degraded5_{itplt}_medium'
    # dataset = f'degraded6_{itplt}_slight'
    csv_file = f'./test_res/{model_name}_{dataset}.csv'
    csv_file = open(csv_file, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(('name', 'psnr', 'niqe', 'ssim', 'lpips'))

    weight_file = '/data0/jli/project/BasicSR-iqa/experiments/pretrained_models/RealESRGAN_x4plus.pth'
    root_dir = f'/data0/jli/datasets/{dataset}/'
    out_root_dir = f'./test_res/{model_name}_{dataset}_metrics/'
    hr_dir = '/data0/jli/datasets/PIPAL/'

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

    model = G().to(device)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['params_ema'])

    model.eval()

    for dir in lr_dirs:
        outputs_dir = out_root_dir + dir + '/'
        lr_dir = root_dir + dir + '/'
        lr_lists = os.listdir(lr_dir)
        Avg_psnr = utils.AverageMeter()
        Avg_niqe = utils.AverageMeter()
        Avg_ssim = utils.AverageMeter()
        Avg_lpips = utils.AverageMeter()
        with tqdm(total=len(lr_lists)) as t:
            t.set_description(f"Processing: {dir}")
            for imgName in lr_lists:
                image = utils.loadIMG_crop(lr_dir + imgName, scale)
                GT = utils.loadIMG_crop(hr_dir + imgName, scale)
                img_mode = image.mode
                if image.mode == 'L':
                    image = image.convert('RGB')
                    GT = GT.convert('RGB')

                lr_image = np.array(image)
                lr = lr_image.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
                lr /= 255.
                lr = torch.from_numpy(lr).to(device).unsqueeze(0)

                with torch.no_grad():
                    SR = model(lr)

                SR = SR.mul(255.0).cpu().numpy().squeeze(0)
                SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])

                SR = np.array(SR).astype(np.uint8)
                SR = Image.fromarray(SR)  # hw -> wh
                SR.save(outputs_dir + imgName)

                metric = metrics.calc_metric(SR, GT, ['psnr', 'ssim', 'niqe', 'lpips'])
                Avg_lpips.update(metric['lpips'], 1)
                Avg_psnr.update(metric['psnr'], 1)
                Avg_niqe.update(metric['niqe'], 1)
                Avg_ssim.update(metric['ssim'], 1)
                t.update(1)

        writer.writerow((dir, Avg_psnr.avg, Avg_niqe.avg, Avg_ssim.avg, Avg_lpips.avg))



