import numpy as np
import h5py
import os
import utils
from imresize import imresize
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def gen_valdata(config):
    scale = config["scale"]
    size_output = config['size_output']
    method = config['method']
    h5savepath = config["hrDir"] + f'_label={size_output}_val_ESRGANx{scale}.h5'
    hrDir = config["hrDir"] + '/'
    h5_file = h5py.File(h5savepath, 'w')
    lr_group = h5_file.create_group('data')
    hr_group = h5_file.create_group('label')
    imgList = os.listdir(hrDir)
    for i, imgName in enumerate(imgList):
        hrIMG = utils.loadIMG_crop(hrDir + imgName, scale).convert('RGB')
        hr = utils.img2ycbcr(hrIMG, gray2rgb=True).astype(np.float32)
        lr = imresize(np.array(hrIMG).astype(np.float32), 1 / scale, method)

        data = lr.astype(np.float32).transpose([2, 0, 1])
        label = hr.transpose([2, 0, 1])

        lr_group.create_dataset(str(i), data=data)
        hr_group.create_dataset(str(i), data=label)
    h5_file.close()


def scale_OST():
    # some OST image has a size lower than 128*128, resize it with scale 2
    root_dir = '../datasets/OST/'

    img_names = os.listdir(root_dir)
    for name in img_names:
        img_path = root_dir + name
        image = Image.open(img_path)

        if image.width < 128 or image.height < 128:
            print(img_path)
            image.save('../datasets/OST_backup/' + name)
            image = np.array(image)
            image = imresize(image, 2, 'bicubic')
            image = Image.fromarray(image.astype(np.uint8))
            image.save(img_path)


if __name__ == '__main__':
    config = {'hrDir': '../datasets/Set14', 'scale': 4, 'size_output': 128, 'method': 'bicubic'}
    gen_valdata(config)
