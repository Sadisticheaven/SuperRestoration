import numpy as np
import h5py
import os
import utils
from PIL import Image
from imresize import imresize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def loadIMG(img_path, scale):
    hrIMG = Image.open(img_path)

    lr_wid = hrIMG.width // scale
    lr_hei = hrIMG.height // scale
    hr_wid = lr_wid * scale
    hr_hei = lr_hei * scale

    hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei))

    if hrIMG.mode == 'L':
        hr = np.array(hrIMG)
        hr = hr.astype(np.float32)
    else:
        hrIMG.convert('RGB')
        hr = np.array(hrIMG)
        hr = utils.rgb2ycbcr(hr).astype(np.float32)[..., 0]
    return hr,

def gen_traindata(config):
    scale = config["scale"]
    stride = config["stride"]
    h5savepath = config["hrDir"] + f'_train_SRCNNx{scale}.h5'
    subimg_savepath = config["hrDir"] + f'train_SRCNNx{scale}/'
    hrDir = config["hrDir"] + '/'
    size_input = config["size_input"]
    size_label = config["size_label"]
    padding = int(abs(size_input - size_label) / 2)

    h5_file = h5py.File(h5savepath, 'w')
    imgList = os.listdir(hrDir)
    lr_subimgs = []
    hr_subimgs = []
    for imgName in imgList:
        # hrIMG = Image.open(hrDir + imgName)
        #
        # lr_wid = hrIMG.width // scale
        # lr_hei = hrIMG.height // scale
        # hr_wid = lr_wid * scale
        # hr_hei = lr_hei * scale
        #
        # hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei))
        #
        # if hrIMG.mode == 'L':
        #     hr = np.array(hrIMG)
        #     hr = hr.astype(np.float32)
        # else:
        #     hrIMG.convert('RGB')
        #     hr = np.array(hrIMG)
        #     hr = utils.rgb2ycbcr(hr).astype(np.float32)[..., 0]

        lr = imresize(hr, 1 / scale, 'bicubic')
        lr = imresize(lr, scale, 'bicubic').astype(np.float32)

        for r in range(0, hr_hei - size_input + 1, stride):
            for c in range(0, hr_wid - size_input + 1, stride):
                lr_subimgs.append(lr[r: r + size_input, c: c + size_input])
                hr_subimgs.append(hr[r + padding: r + padding + size_label, c + padding: c + padding + size_label])

    lr_subimgs = np.array(lr_subimgs)
    hr_subimgs = np.array(hr_subimgs)

    h5_file.create_dataset('data', data=lr_subimgs)
    h5_file.create_dataset('label', data=hr_subimgs)

    h5_file.close()


def gen_valdata(config):
    scale = config["scale"]
    stride = config["stride"]
    h5savepath = config["hrDir"] + f'_val_SRCNNx{scale}.h5'
    subimg_savepath = config["hrDir"] + f'_val_SRCNNx{scale}/'
    hrDir = config["hrDir"] + '/'
    size_input = config["size_input"]
    size_label = config["size_label"]
    padding = int(abs(size_input - size_label) / 2)
    # utils.mkdirs(subimg_savepath)

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

        hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei))

        if hrIMG.mode == 'L':
            hr = np.array(hrIMG)
            hr = hr.astype(np.float32)
        else:
            hrIMG.convert('RGB')
            hr = np.array(hrIMG)
            hr = utils.rgb2ycbcr(hr).astype(np.float32)[..., 0]

        data = imresize(hr, 1 / scale, 'bicubic')
        data = imresize(data, scale, 'bicubic')
        data = data.astype(np.float32)
        label = hr[padding: -padding, padding: -padding]
        lr_group.create_dataset(str(i), data=data)
        hr_group.create_dataset(str(i), data=label)

    h5_file.close()


if __name__ == '__main__':
    # config = {'hrDir': '../datasets/T91_aug', 'scale': 3, "stride": 14, "size_input": 33, "size_label": 21}
    config = {'hrDir': '../datasets/T91', 'scale': 2, "stride": 14, "size_input": 22, "size_label": 10}
    gen_traindata(config)
    config['hrDir'] = '../datasets/Set5'
    gen_valdata(config)
