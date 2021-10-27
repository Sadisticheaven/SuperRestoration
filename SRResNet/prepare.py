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
    size_input = config["size_input"]
    size_label = size_input * scale
    size_output = config['size_output']
    method = config['method']
    h5savepath = config["hrDir"] + f'_label={size_output}_train_SRResNetx{scale}.h5'
    hrDir = config["hrDir"] + '/'


    h5_file = h5py.File(h5savepath, 'w')
    imgList = os.listdir(hrDir)
    lr_subimgs = []
    hr_subimgs = []
    for imgName in imgList:
        tpath = os.path.join(hrDir + imgName)
        hrIMG = Image.open(tpath)

        lr_wid = hrIMG.width // scale
        lr_hei = hrIMG.height // scale
        hr_wid = lr_wid * scale
        hr_hei = lr_hei * scale

        hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei)).convert('RGB')
        hr = np.array(hrIMG)
        hr = utils.rgb2ycbcr(hr).astype(np.float32)

        lr = imresize(hr, 1 / scale, method)
        input = lr.astype(np.float32)

        for r in range(0, lr_hei - size_input + 1, stride):
            for c in range(0, lr_wid - size_input + 1, stride):
                lr_subimgs.append(input[r: r + size_input, c: c + size_input])
                label = hr[r * scale: r * scale + size_label, c * scale: c * scale + size_label]
                hr_subimgs.append(label)

    lr_subimgs = np.array(lr_subimgs).astype(np.float32)
    h5_file.create_dataset('data', data=lr_subimgs)
    lr_subimgs = []
    # HR dataset is too large to convert to ndarray, so convert it by several parts.
    num = len(hr_subimgs)
    seg = num // scale
    hr_imgs = np.array(hr_subimgs[0: seg]).astype(np.float32)
    dl = h5_file.create_dataset('label', data=hr_imgs, maxshape=[num, 96, 96, 3])
    for i in range(1, scale):
        hr_imgs = []
        hr_imgs = np.array(hr_subimgs[i*seg: (i+1)*seg]).astype(np.float32)
        dl.resize(((i+1)*seg, 96, 96, 3))
        dl[i*seg:] = hr_imgs
    hr_imgs = []
    hr_imgs = np.array(hr_subimgs[scale*seg: num]).astype(np.float32)
    dl.resize((num, 96, 96, 3))
    dl[scale*seg:] = hr_imgs
    hr_imgs = []
    h5_file.close()


def gen_valdata(config):
    scale = config["scale"]
    size_input = config["size_input"]
    size_label = size_input * scale
    size_output = config['size_output']
    method = config['method']
    h5savepath = config["hrDir"] + f'_label={size_output}_val_SRResNetx{scale}.h5'
    hrDir = config["hrDir"] + '/'
    h5_file = h5py.File(h5savepath, 'w')
    lr_group = h5_file.create_group('data')
    hr_group = h5_file.create_group('label')
    imgList = os.listdir(hrDir)
    for i, imgName in enumerate(imgList):
        tpath = os.path.join(hrDir + imgName)
        hrIMG = Image.open(tpath)

        lr_wid = hrIMG.width // scale
        lr_hei = hrIMG.height // scale
        hr_wid = lr_wid * scale
        hr_hei = lr_hei * scale

        hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei)).convert('RGB')
        hr = np.array(hrIMG)
        hr = utils.rgb2ycbcr(hr).astype(np.float32)

        lr = imresize(hr, 1 / scale, method)
        data = lr.astype(np.float32)
        # residual
        label = hr.astype(np.float32)

        lr_group.create_dataset(str(i), data=data)
        hr_group.create_dataset(str(i), data=label)

    h5_file.close()

if __name__ == '__main__':
    # config = {'hrDir': './test/flower', 'scale': 3, "stride": 14, "size_input": 33, "size_label": 21}
    config = {'hrDir': '../datasets/291_aug', 'scale': 4, 'stride': 12, "size_input": 24, "size_output": 96, 'method': 'bicubic'}
    gen_traindata(config)
    config['hrDir'] = '../datasets/Set5'
    gen_valdata(config)
