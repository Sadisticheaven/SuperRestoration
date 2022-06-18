import csv
from tqdm import tqdm
import utils
import metrics
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'RealESRGAN'
    scale = 4
    itplt = 'bilinear'
    # itplt = 'bicubic'
    dataset = f'degraded4_{itplt}_heavy'
    # dataset = f'degraded5_{itplt}_medium'
    # dataset = f'degraded6_{itplt}_slight'

    sr_root_dir_degradation = f'./test_res/{model_name}_{dataset}/sort_with_degradation/'
    gt_dir = './datasets/PIPAL/'
    deg_types = os.listdir(sr_root_dir_degradation)

    csv_file = f'./test_res/{model_name}_{dataset}.csv'
    csv_file = open(csv_file, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(('name', 'psnr', 'niqe', 'ssim', 'lpips'))
    for deg_type in deg_types:
        sr_dir = sr_root_dir_degradation + deg_type + '/'
        sr_lists = os.listdir(sr_dir)
        Avg_psnr = utils.AverageMeter()
        Avg_niqe = utils.AverageMeter()
        Avg_ssim = utils.AverageMeter()
        Avg_lpips = utils.AverageMeter()
        with tqdm(total=len(sr_lists)) as t:
            t.set_description(f"Processing: {deg_type}")
            for imgName in sr_lists:
                SR = utils.loadIMG_crop(sr_dir + imgName, scale)
                GT = utils.loadIMG_crop(gt_dir + imgName, scale)
                my_metrics = metrics.calc_metric(SR, GT, ['lpips'])
                Avg_lpips.update(my_metrics['lpips'], 1)
                # Avg_psnr.update(my_metrics['psnr'], 1)
                # Avg_niqe.update(my_metrics['niqe'], 1)
                # Avg_ssim.update(my_metrics['ssim'], 1)
                t.update(1)
            # writer.writerow((deg_type, Avg_psnr.avg, Avg_niqe.avg, Avg_ssim.avg, Avg_lpips.avg))
