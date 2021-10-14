import numpy as np
from torchvision import transforms
from PIL import Image
import os
import utils
from imresize import imresize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    rootdir = r'./datasets/T91/'  # 指明被遍历的文件夹
    savePath = './datasets/T91_aug/'
    utils.mkdirs(savePath)
    imgList = os.listdir(rootdir)
    for imgName in imgList:
        tpath = os.path.join(rootdir + imgName)
        spath = os.path.splitext(imgName)
        fopen = Image.open(tpath)
        for angle in list({0, 90, 180, 270}):
            img = fopen.rotate(angle, expand=True)
            name = savePath + spath[0] + '_' + str(angle)
            img.save(name + '_x1.0.bmp')  # + spath[1])
            img = np.array(img)
            for scale in list({0.6, 0.7, 0.8, 0.9}):
                res = imresize(img, scale, 'bicubic')
                res = Image.fromarray(res.astype(np.uint8))
                res.save(name + '_x' + str(scale) + '.bmp')