import h5py
import torch
from scipy.io import loadmat
import utils
import numpy as np
import transplant
import os
from PIL import Image
import imresize
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'




if __name__ == '__main__':
    # h5_path = '../datasets/Matlab_Set5_val_SRCNNx3.h5'
    # with h5py.File(h5_path, 'r') as f:
    #     len = len(f['data'])
    #     for idx in range(len):
    #         input = f['data'][idx]
    #         label = f['label'][idx]

    # m = loadmat("G:/Document/文献资料/超分辨率/SRCNN_train/SRCNN/model/9-1-5(91 images)/x2.mat")
    # x = m['weights_conv1']
    # x = np.reshape(x, (9, 9, 64))
    # ax = utils.viz_layer2(x, 64)

    hrfile = 'G:/Document/pythonProj/SuperRestoration/datasets/T91_aug/t10_0_x0.6.png'
    hrfile2 = 'G:/Document/pythonProj/SuperRestoration/datasets/T91_augt10_rot0_s6.bmp'
    matlab = transplant.Matlab(executable='G:/Software/Matlab2020b/bin/matlab.exe')
    hr_py = Image.open(hrfile).convert('RGB')  # RGBA->RGB
    hr_py = np.array(hr_py)
    hr_mat = matlab.imread(hrfile)
    hr_mat = hr_mat[0][:, :, :]
    diff = hr_py - hr_mat  # 无差别
    hr_mat = matlab.rgb2ycbcr(hr_py)
    hr_mat_y = hr_mat[:, :, 0]
    hr_py_ycbcr = utils.rgb2ycbcr(hr_py)
    hr_py_ycbcr = hr_py_ycbcr.astype(np.uint8)
    hr_py_y = hr_py_ycbcr[:, :, 0]
    diff = hr_mat - hr_py_ycbcr  # 有很大差别，所以不能使用python的ycbcr

    hr_py = Image.open(hrfile).convert('RGB')
    hr_mat = matlab.imread(hrfile)
    hr_mat = hr_mat[0][:, :, :]
    scale = 3
    hr_py = np.array(hr_py).astype(np.uint8)
    lr_mat = matlab.imresize(hr_py, 1 / scale, 'bicubic')[0]
    lr_mat2 = imresize.imresize(hr_py, 1 / scale)
    diff = lr_mat2 - lr_mat


