import numpy as np
import options
import torch
from torch.backends import cudnn
import utils
from SRFlow_model import SRFlowModel
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'SRFlow'
    weight_file = '../weight_file/SRFlow_DF2K_4X.pth'
    root_dir = '../datasets/degraded9/'
    out_root_dir = f'../test_res/{model_name}_degraded4/'
    hr_dir = '../datasets/PIPAL/'

    lr_dirs = os.listdir(root_dir)
    # 根据图片名称分类输出
    out_dirs = os.listdir(hr_dir)
    for dir in out_dirs:
        dir = dir.split('.')[0] + '/'
        utils.mkdirs(out_root_dir + dir)

    scale = 4
    padding = scale

    if not os.path.exists(weight_file):
        print(f'Weight file not exist!\n{weight_file}\n')
        raise "Error"

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = options.parse('./SRFlow_DF2K_4X.yml', is_train=False)
    opt['gpu_ids'] = None
    opt = options.dict_to_nonedict(opt)
    heat = opt['heat']
    model = SRFlowModel(opt)
    checkpoint = torch.load(weight_file)
    model.netG.load_state_dict(checkpoint)
    offset = 0
    for dir in lr_dirs:
        # outputs_dir = out_root_dir + dir + '/'
        lr_dir = root_dir + dir + '/'
        lr_lists = os.listdir(lr_dir)
        Avg_psnr = utils.AverageMeter()
        Avg_niqe = utils.AverageMeter()
        for imgName in lr_lists:
            outputs_dir = out_root_dir + imgName.split('.')[0] + '/'
            image = utils.loadIMG_crop(lr_dir + imgName, scale)
            image = utils.ImgOffSet(image, offset, offset)
            hr_image = utils.loadIMG_crop(hr_dir + imgName, scale)
            hr_image = utils.ImgOffSet(hr_image, offset*scale, offset*scale)

            img_mode = image.mode
            if img_mode == 'L':
                gray_img = np.array(image)
            image = image.convert('RGB')
            lr_image = np.array(image)
            hr_image = np.array(hr_image)

            lr = lr_image.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
            lr /= 255.
            lr = torch.from_numpy(lr).unsqueeze(0)

            # hr_image = hr_image[padding: -padding, padding: -padding, ...]

            SR = model.get_sr(lr, heat)
            # SR = SR[..., padding: -padding, padding: -padding]
            SR = SR.mul(255.0).cpu().numpy().squeeze(0)
            SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])
            if img_mode != 'L':
                SR_y = utils.rgb2ycbcr(SR).astype(np.float32)[..., 0] / 255.
                hr_y = utils.rgb2ycbcr(hr_image).astype(np.float32)[..., 0] / 255.
            else:
                # gray_img = gray_img.astype(np.float32)[padding: -padding, padding: -padding, ...]
                hr_y = gray_img / 255.
                SR = Image.fromarray(SR.astype(np.uint8)).convert('L')
                SR_y = np.array(SR).astype(np.float32) / 255.
            psnr = utils.calc_psnr(hr_y, SR_y)
            # NIQE = niqe.calculate_niqe(SR_y)
            Avg_psnr.update(psnr, 1)
            # Avg_niqe.update(NIQE, 1)
            print(f'{imgName}, ' + 'PSNR: {:.2f}'.format(psnr.item()))
            # print(f'{imgName}, ' + 'PSNR: {:.2f} , NIQE: {:.4f}'.format(psnr.item(), NIQE))
            # GPU tensor -> CPU tensor -> numpy
            output = np.array(SR).astype(np.uint8)
            output = Image.fromarray(output)  # hw -> wh
            # output.save(outputs_dir + imgName)
            output.save(outputs_dir + dir + '.bmp')
            # output.save(outputs_dir + imgName.replace('.', f'_{model_name}_x{scale}.'))
        print('Average_PSNR: {:.2f}, Average_NIQE: {:.4f}'.format(Avg_psnr.avg, Avg_niqe.avg))



