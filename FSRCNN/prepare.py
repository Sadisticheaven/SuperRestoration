import numpy as np
import h5py
import os
import utils
from PIL import Image
from imresize import imresize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def gen_traindata(config):
    scale = config["scale"]
    stride = config["stride"]
    if config['residual']:
        h5savepath = config["hrDir"] + f'_train_FSRCNNx{scale}_res.h5'
    else:
        h5savepath = config["hrDir"] + f'_train_FSRCNNx{scale}.h5'
    hrDir = config["hrDir"] + '/'
    size_input = config["size_input"]
    size_label = size_input * scale
    size_output = config['size_output']
    padding = (size_label - size_output) // 2

    h5_file = h5py.File(h5savepath, 'w')
    imgList = os.listdir(hrDir)
    lr_subimgs = []
    hr_subimgs = []
    for imgName in imgList:
        tpath = os.path.join(hrDir + imgName)
        hrIMG = Image.open(tpath).convert('RGB')

        lr_wid = hrIMG.width // scale
        lr_hei = hrIMG.height // scale
        hr_wid = lr_wid * scale
        hr_hei = lr_hei * scale

        hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei))
        hr = np.array(hrIMG)
        hr_y = utils.rgb2ycbcr(hr).astype(np.float32)[..., 0]

        lr_y = imresize(hr_y, 1 / scale, 'bicubic')
        input = lr_y.astype(np.float32)
        # residual
        if config['residual']:
            input_upscale = imresize(input, scale, 'bicubic')
            input_upscale = input_upscale.astype(np.float32)
            hr_y = hr_y - input_upscale

        for r in range(0, lr_hei - size_input + 1, stride):
            for c in range(0, lr_wid - size_input + 1, stride):
                lr_subimgs.append(lr_y[r: r + size_input, c: c + size_input])
                label = hr_y[r * scale: r * scale + size_label, c * scale: c * scale + size_label]
                label = label[padding: -padding, padding: -padding]
                hr_subimgs.append(label)

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
    if config['residual']:
        h5savepath = config["hrDir"] + f'_val_FSRCNNx{scale}_res.h5'
    else:
        h5savepath = config["hrDir"] + f'_val_FSRCNNx{scale}.h5'
    hrDir = config["hrDir"] + '/'
    h5_file = h5py.File(h5savepath, 'w')
    lr_group = h5_file.create_group('data')
    hr_group = h5_file.create_group('label')
    imgList = os.listdir(hrDir)
    for i, imgName in enumerate(imgList):
        tpath = os.path.join(hrDir + imgName)
        hrIMG = Image.open(tpath).convert('RGB')

        lr_wid = hrIMG.width // scale
        lr_hei = hrIMG.height // scale
        hr_wid = lr_wid * scale
        hr_hei = lr_hei * scale

        hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei))
        hr = np.array(hrIMG)
        hr_y = utils.rgb2ycbcr(hr).astype(np.float32)[..., 0]

        lr_y = imresize(hr_y, 1 / scale, 'bicubic')
        data = lr_y.astype(np.float32)
        # residual
        if config['residual']:
            input_upscale = imresize(input, scale, 'bicubic')
            input_upscale = input_upscale.astype(np.float32)
            label = hr_y - input_upscale
        else:
            label = hr_y
        label = label.astype(np.float32)[padding: -padding, padding: -padding]

        lr_group.create_dataset(str(i), data=data)
        hr_group.create_dataset(str(i), data=label)

    h5_file.close()

if __name__ == '__main__':
    # config = {'hrDir': './test/flower', 'scale': 3, "stride": 14, "size_input": 33, "size_label": 21}
    config = {'hrDir': '../datasets/T91_aug', 'scale': 3, 'stride': 10, "size_input": 11, "size_output": 19, "residual": False}
    gen_traindata(config)
    config['hrDir'] = '../datasets/Set5'
    gen_valdata(config)
