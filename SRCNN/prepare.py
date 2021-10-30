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
    h5savepath = config["hrDir"] + f'_train_SRCNNx{scale}.h5'
    hrDir = config["hrDir"] + '/'
    size_input = config["size_input"]
    size_label = config["size_label"]
    padding = int(abs(size_input - size_label) / 2)

    h5_file = h5py.File(h5savepath, 'w')
    imgList = os.listdir(hrDir)
    lr_subimgs = []
    hr_subimgs = []
    with tqdm(total=len(imgList)) as t:
        for imgName in imgList:
            hrIMG = utils.loadIMG_crop(hrDir + imgName, scale)
            hr_y = utils.img2ycbcr(hrIMG)[..., 0]
            lr = imresize(hr_y, 1 / scale, 'bicubic')
            lr = imresize(lr, scale, 'bicubic').astype(np.float32)

            for r in range(0, hr_y.shape[0] - size_input + 1, stride):  # hr.height
                for c in range(0, hr_y.shape[1] - size_input + 1, stride):  # hr.width
                    lr_subimgs.append(lr[r: r + size_input, c: c + size_input])
                    hr_subimgs.append(hr_y[r + padding: r + padding + size_label, c + padding: c + padding + size_label])
            t.update(1)

    lr_subimgs = np.array(lr_subimgs)
    hr_subimgs = np.array(hr_subimgs)

    h5_file.create_dataset('data', data=lr_subimgs)
    h5_file.create_dataset('label', data=hr_subimgs)

    h5_file.close()


def gen_valdata(config):
    scale = config["scale"]
    h5savepath = config["hrDir"] + f'_val_SRCNNx{scale}.h5'
    hrDir = config["hrDir"] + '/'
    size_input = config["size_input"]
    size_label = config["size_label"]
    padding = int(abs(size_input - size_label) / 2)

    h5_file = h5py.File(h5savepath, 'w')
    lr_group = h5_file.create_group('data')
    hr_group = h5_file.create_group('label')
    imgList = os.listdir(hrDir)
    for i, imgName in enumerate(imgList):
        hrIMG = utils.loadIMG_crop(hrDir + imgName, scale)
        hr_y = utils.img2ycbcr(hrIMG)[..., 0]

        data = imresize(hr_y, 1 / scale, 'bicubic')
        data = imresize(data, scale, 'bicubic')
        data = data.astype(np.float32)
        label = hr_y[padding: -padding, padding: -padding]
        lr_group.create_dataset(str(i), data=data)
        hr_group.create_dataset(str(i), data=label)

    h5_file.close()


if __name__ == '__main__':
    # config = {'hrDir': '../datasets/T91_aug', 'scale': 3, "stride": 14, "size_input": 33, "size_label": 21}
    config = {'hrDir': '../datasets/T91_aug', 'scale': 2, "stride": 14, "size_input": 22, "size_label": 10}
    gen_traindata(config)
    config['hrDir'] = '../datasets/Set5'
    gen_valdata(config)
