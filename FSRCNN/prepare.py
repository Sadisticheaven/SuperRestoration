import numpy as np
import h5py
import os
from tqdm import tqdm
import utils
from imresize import imresize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def gen_traindata(config):
    scale = config["scale"]
    stride = config["stride"]
    size_input = config["size_input"]
    size_label = size_input * scale
    size_output = config['size_output']
    padding = abs(size_label - size_output) // 2
    if scale == 3:
        padding2 = padding
    else:
        padding2 = padding + 1
    method = config['method']
    if config['residual']:
        h5savepath = config["hrDir"] + f'_label={size_output}_train_FSRCNNx{scale}_res.h5'
    else:
        h5savepath = config["hrDir"] + f'_label={size_output}_train_FSRCNNx{scale}.h5'
    hrDir = config["hrDir"] + '/'
    h5_file = h5py.File(h5savepath, 'w')
    imgList = os.listdir(hrDir)
    lr_subimgs = []
    hr_subimgs = []
    with tqdm(total=len(imgList)) as t:
        for imgName in imgList:
            hrIMG = utils.loadIMG_crop(hrDir + imgName, scale)
            hr_y = utils.img2ycbcr(hrIMG)[..., 0]
            lr_y = imresize(hr_y, 1 / scale, method).astype(np.float32)
            # residual
            if config['residual']:
                lr_y_upscale = imresize(lr_y, scale, method).astype(np.float32)
                hr_y = hr_y - lr_y_upscale

            for r in range(0, lr_y.shape[0] - size_input + 1, stride):
                for c in range(0, lr_y.shape[1] - size_input + 1, stride):
                    lr_subimgs.append(lr_y[r: r + size_input, c: c + size_input])
                    label = hr_y[r * scale: r * scale + size_label, c * scale: c * scale + size_label]
                    label = label[padding: -padding2, padding: -padding2]
                    hr_subimgs.append(label)
            t.update(1)

    lr_subimgs = np.array(lr_subimgs).astype(np.float32)
    hr_subimgs = np.array(hr_subimgs).astype(np.float32)

    h5_file.create_dataset('data', data=lr_subimgs)
    h5_file.create_dataset('label', data=hr_subimgs)

    h5_file.close()


def gen_valdata(config):
    scale = config["scale"]
    size_input = config["size_input"]
    size_label = size_input * scale
    size_output = config['size_output']
    padding = (size_label - size_output) // 2
    method = config['method']
    if scale == 3:
        padding2 = padding
    else:
        padding2 = padding + 1
    if config['residual']:
        h5savepath = config["hrDir"] + f'_label={size_output}_val_FSRCNNx{scale}_res.h5'
    else:
        h5savepath = config["hrDir"] + f'_label={size_output}_val_FSRCNNx{scale}.h5'
    hrDir = config["hrDir"] + '/'
    h5_file = h5py.File(h5savepath, 'w')
    lr_group = h5_file.create_group('data')
    hr_group = h5_file.create_group('label')
    if config['residual']:
        bic_group = h5_file.create_group('bicubic')
    imgList = os.listdir(hrDir)
    for i, imgName in enumerate(imgList):
        hrIMG = utils.loadIMG_crop(hrDir+imgName, scale)
        hr_y = utils.img2ycbcr(hrIMG)[..., 0]

        lr_y = imresize(hr_y, 1 / scale, method).astype(np.float32)
        # residual
        if config['residual']:
            bic_y = imresize(lr_y, scale, method).astype(np.float32)
            bic_y = bic_y[padding: -padding2, padding: -padding2]
        label = hr_y.astype(np.float32)[padding: -padding2, padding: -padding2]

        lr_group.create_dataset(str(i), data=lr_y)
        hr_group.create_dataset(str(i), data=label)
        if config['residual']:
            bic_group.create_dataset(str(i), data=bic_y)
    h5_file.close()

if __name__ == '__main__':
    # config = {'hrDir': './test/flower', 'scale': 3, "stride": 14, "size_input": 33, "size_label": 21}
    config = {'hrDir': '../datasets/T91_aug', 'scale': 4, 'stride': 10, "size_input": 10, "size_output": 21, "residual": True, 'method': 'bicubic'}
    gen_traindata(config)
    config['hrDir'] = '../datasets/Set5'
    gen_valdata(config)
