# PyTorch
import cv2
import torch
import utils
import numpy as np
import lpips
import niqe
calc_lpips = lpips.LPIPS(net='alex', model_path='./weight_file/alexnet-owt-7be5be79.pth')


def calc_psnr(img1, img2):
    if isinstance(img1, torch.Tensor):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
    else:
        return 10. * np.log10(1. / np.mean((img1 - img2) ** 2))


def calculate_ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calc_metric(sr_image, gt_image, metric, crop=0):
    """
    Args:
        sr_image: low quality img, [r,g,b] or gray 【0,255】
        gt_image: high quality img, [r,g,b] or gray (same size with SR)
        metric: string array, 'psnr','ssim','niqe'
        crop: crop pixel of border
    Returns:
        dict of matric with correspond float value
    """

    res = dict()
    # convert RGBA to RGB
    if sr_image.mode != 'L':
        sr_image = sr_image.convert('RGB')
        gt_image = gt_image.convert('RGB')
    # crop border
    sr = np.array(sr_image).astype(np.float32)[crop: -max(crop, 1), crop: -max(crop, 1), ...]
    gt = np.array(gt_image).astype(np.float32)[crop: -max(crop, 1), crop: -max(crop, 1), ...]
    # RGB to YCbCr
    if sr_image.mode == 'L':
        sr_y = sr / 255.
        gt_y = gt / 255.
    else:
        sr_y = utils.rgb2ycbcr(sr).astype(np.float32)[..., 0] / 255.
        gt_y = utils.rgb2ycbcr(gt).astype(np.float32)[..., 0] / 255.
    # calculate matrics
    if 'psnr' in metric:
        res.update({'psnr': calc_psnr(sr_y, gt_y)})
    if 'niqe' in metric:
        res.update({'niqe': niqe.calculate_niqe(sr_y)})
    if 'ssim' in metric:
        res.update({'ssim': calculate_ssim(sr_y * 255, gt_y * 255)})
    if 'lpips' in metric:
        res.update({'lpips': calc_lpips(
            torch.from_numpy((sr / 255.).transpose([2, 0, 1])).unsqueeze(0),
            torch.from_numpy((gt / 255.).transpose([2, 0, 1])).unsqueeze(0),
            normalize=True
        ).item()})
    return res

def test_calc_metric():
    SR = utils.loadIMG_crop('./test_res/HCFlowSR/bicubicx4/sample=0/A0012.bmp', 4).convert('RGB')
    GT = utils.loadIMG_crop('./datasets/PIPAL/A0012.bmp', 4).convert('RGB')
    metric = calc_metric(SR, GT, ['psnr', 'ssim', 'niqe', 'lpips'])

if __name__ == '__main__':
    test_calc_metric()


