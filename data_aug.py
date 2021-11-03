import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import utils
from imresize import imresize
import threading
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def T91_aug():
    rootdir = r'./datasets/T91/'  # 指明被遍历的文件夹
    savePath = './datasets/T91_aug/'
    utils.mkdirs(savePath)
    imgList = os.listdir(rootdir)
    total = len(imgList)
    with tqdm(total=total) as t:
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
            t.update(1)


def T291_aug():
    rootdir = r'./datasets/291/'  # 指明被遍历的文件夹
    savePath = './datasets/291_aug/'
    utils.mkdirs(savePath)
    imgList = os.listdir(rootdir)
    total = len(imgList)
    with tqdm(total=total) as t:
        for imgName in imgList:
            tpath = os.path.join(rootdir + imgName)
            spath = os.path.splitext(imgName)

            fopen = Image.open(tpath)
            flipimgs = [fopen,
                        fopen.transpose(Image.FLIP_LEFT_RIGHT)
                        ]
            for i, flipimg in enumerate(flipimgs):
                for angle in list({0, 90, 180, 270}):
                    img = flipimg.rotate(angle, expand=True)
                    name = savePath + spath[0] + f'_{i}_' + str(angle)
                    img.save(name + '_x1.0.bmp')  # + spath[1])
                    img = np.array(img)
                    for scale in list({0.5, 0.7}):
                        res = imresize(img, scale, 'bicubic')
                        res = Image.fromarray(res.astype(np.uint8))
                        res.save(name + '_x' + str(scale) + '.bmp')
            t.update(1)


class myThread(threading.Thread):
    def __init__(self, img_list, t):
        threading.Thread.__init__(self)
        self.img_list = img_list
        self.t = t

    def run(self):
        print("Start thread： " + self.name)
        augment(self.img_list, self.t)


def augment(img_list, t):
    for imgName in img_list:
        spath = os.path.splitext(imgName)
        fopen = Image.open(rootdir + imgName)
        for angle in list({0, 90, 180, 270}):
            img = fopen.rotate(angle, expand=True)
            name = savePath + spath[0] + '_' + str(angle)
            img.save(name + '_x1.0.bmp')  # + spath[1])
            img = np.array(img)
            for scale in list({0.6, 0.7, 0.8, 0.9}):
                res = imresize(img, scale, 'bicubic')
                res = Image.fromarray(res.astype(np.uint8))
                res.save(name + '_x' + str(scale) + '.bmp')
        threadLock.acquire()
        t.update(1)
        threadLock.release()


def DIV2K_aug():
    imgList = os.listdir(rootdir)
    total = len(imgList)
    seg = total // thread_num
    with tqdm(total=total) as t:
        for i in range(thread_num):
            if i == thread_num-1:
                thread = myThread(imgList[seg * i: total], t)
            else:
                thread = myThread(imgList[seg*i: seg*(i+1)], t)
            thread.start()
            threads.append(thread)
        for th in threads:
            th.join()


threadLock = threading.Lock()
rootdir = r'./datasets/T91/'  # 指明被遍历的文件夹
savePath = './datasets/DIV2K_aug/'
utils.mkdirs(savePath)
threads = []
thread_num = 8

if __name__ == '__main__':
    DIV2K_aug()
