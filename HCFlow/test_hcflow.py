import numpy as np
from tqdm import tqdm

import options
import torch
from torch.backends import cudnn
import utils
from HCFlow_SR_model import HCFlowSRModel
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    model_name = 'HCFlowSR'
    weight_file = '../weight_file/SR_DF2K_X4_HCFlow++.pth'
    root_dir = '../datasets/degraded_srflow/'
    out_root_dir = f'../test_res/{model_name}/'
    hr_dir = '../datasets/PIPAL/'

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
    opt = options.parse('./test_SR_DF2K_4X_HCFlow.yml', is_train=False)
    opt['gpu_ids'] = '0'
    opt = options.dict_to_nonedict(opt)
    heat = 0.8
    model = HCFlowSRModel(opt, [heat])
    checkpoint = torch.load(weight_file)
    model.netG.load_state_dict(checkpoint)
    offset = 0

    for dir in lr_dirs:
        outputs_dir = out_root_dir + dir + '/'
        lr_dir = root_dir + dir + '/'
        lr_lists = os.listdir(lr_dir)
        with tqdm(total=len(lr_lists)) as t:
            t.set_description(f"Processing: {dir}")
            for imgName in lr_lists:
                # outputs_dir = out_root_dir + imgName.split('.')[0] + '/'
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

                data = {'LQ': lr}
                model.feed_data(data, need_GT=False)
                model.test()
                visuals = model.get_current_visuals(need_GT=False)
                for sample in range(len(visuals)-1):
                    SR = visuals['SR', heat, sample]
                    SR = SR.mul(255.0).cpu().numpy()
                    SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])

                    # GPU tensor -> CPU tensor -> numpy
                    output = np.array(SR).astype(np.uint8)
                    output = Image.fromarray(output)  # hw -> wh
                    tmp = outputs_dir + f'sample={sample}/'
                    utils.mkdirs(tmp)
                    output.save(tmp + imgName)
                t.update(1)




