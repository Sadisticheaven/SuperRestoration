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
    def __init__(self, img_list, t, func):
        threading.Thread.__init__(self)
        self.img_list = img_list
        self.t = t
        self.func = func

    def run(self):
        print("Start thread： " + self.name)
        self.func(self.img_list, self.t)


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


def extract_subimg(img_list, t):
    for imgName in img_list:
        hrIMG = utils.loadIMG_crop(rootdir + imgName, scale)
        hr = np.array(hrIMG.convert('RGB'))
        lr = imresize(hr, 1 / scale, 'bicubic')
        input = lr.astype(np.float32)

        for r in range(0, lr.shape[0] - size_input + 1, stride):  # height
            for c in range(0, lr.shape[1] - size_input + 1, stride):  # width
                lr_subimgs = input[r: r + size_input, c: c + size_input]  # hwc -> chw
                lr_subimgs = Image.fromarray(lr_subimgs.astype(np.uint8))
                lr_subimgs.save(lrpath + imgName.replace('.', f'_r{r}_c{c}.'))
                label = hr[r * scale: r * scale + size_label, c * scale: c * scale + size_label]
                label = Image.fromarray(label.astype(np.uint8))
                label.save(hrpath + imgName.replace('.', f'_r{r}_c{c}.'))
        t.update(1)


def DIV2K_aug():
    imgList = os.listdir(rootdir)
    total = len(imgList)
    seg = total // thread_num
    with tqdm(total=total) as t:
        for i in range(thread_num):
            if i == thread_num-1:
                thread = myThread(imgList[seg * i: total], t, augment)
            else:
                thread = myThread(imgList[seg*i: seg*(i+1)], t, augment)
            thread.start()
            threads.append(thread)
        for th in threads:
            th.join()


def DF2KOST_LR():
    imgList = []
    for root_dir in rootdirs:
        img_names = os.listdir(root_dir)
        for name in img_names:
            imgList.append([root_dir, name])
    total = len(imgList)
    seg = total // thread_num
    with tqdm(total=total) as t:
        for i in range(thread_num):
            if i == thread_num-1:
                thread = myThread(imgList[seg * i: total], t, gen_HRLR)
            else:
                thread = myThread(imgList[seg*i: seg*(i+1)], t, gen_HRLR)
            thread.start()
            threads.append(thread)
        for th in threads:
            th.join()

def gen_HRLR(img_list, t):
    for dir, imgName in img_list:
        hrIMG = utils.loadIMG_crop(dir + imgName, scale)
        hr = np.array(hrIMG.convert('RGB'))
        lr = imresize(hr, 1 / scale, 'bicubic')
        lr = Image.fromarray(lr.astype(np.uint8))
        lr.save(lrpath + imgName)
        hr = Image.fromarray(hr.astype(np.uint8))
        hr.save(hrpath + imgName)
        t.update(1)

threadLock = threading.Lock()
scale = 4
size_input = 24
size_label = 96
stride = 12
rootdir = r'./datasets/Set5/'  # 指明被遍历的文件夹
# rootdirs = ['./datasets/Set5/']
rootdirs = ['../datasets/DIV2K_train_HR/', '../datasets/Flickr2K/Flickr2K_HR/', '../datasets/OST/']
savePath = './datasets/DIV2K_aug/'
hrpath = f'./datasets/DF2K+OST/HR/'
lrpath = f'./datasets/DF2K+OST/LRx{scale}/'
utils.mkdirs(savePath)
utils.mkdirs(hrpath)
utils.mkdirs(lrpath)
threads = []
thread_num = 8

if __name__ == '__main__':
    # DIV2K_aug()
    DF2KOST_LR()