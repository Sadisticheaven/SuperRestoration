import os

import lmdb
from PIL import Image
import numpy as np
import albumentations as A
from imresize import imresize
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

class ESRGANTrainDataset(Dataset):
    def __init__(self, root_dirs=['../datasets/DIV2K_train_HR/',
                                  '../datasets/Flickr2K/Flickr2K_HR/',
                                  '../datasets/OST/'
                                  ]):
        super(ESRGANTrainDataset, self).__init__()
        self.data = []

        for root_dir in root_dirs:
            self.img_names = os.listdir(root_dir)
            for name in self.img_names:
                self.data.append(root_dir + name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]

        image = np.array(Image.open(img_path))
        transpose = A.Compose([
            A.RandomCrop(width=128, height=128),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        image = transpose(image=image)["image"]
        label = torch.from_numpy(image.astype(np.float32).transpose([2, 0, 1]) / 255.)
        data = imresize(image, 1 / 4, 'bicubic')
        data = torch.from_numpy(data.astype(np.float32).transpose([2, 0, 1]) / 255.)
        return data, label


class ESRGANValDataset(Dataset):
    def __init__(self, h5_file):
        super(ESRGANValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['data'][str(idx)][:, :, :] / 255.), \
                   torch.from_numpy(f['label'][str(idx)][:, :, :] / 255.)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class lmdbDataset(Dataset):
    def __init__(self):
        super(lmdbDataset, self).__init__()
        env = lmdb.open('../datasets/Set5.lmdb', max_dbs=2, readonly=True)
        self.data = env.open_db("train_data".encode('ascii'))
        self.shape = env.open_db("train_shape".encode('ascii'))
        self.txn = env.begin()
        self._length = int(self.txn.stat(db=self.data)["entries"] / 2)

    def __getitem__(self, idx):
        idx = str(idx)
        image = self.txn.get(idx.encode('ascii'), db=self.data)
        image = np.frombuffer(image, 'uint8')
        buf_meta = self.txn.get((idx+'.meta').encode('ascii'), db=self.shape)
        buf_meta = buf_meta.decode('ascii')
        H, W, C = [int(s) for s in buf_meta.split(',')]
        image = image.reshape(H, W, C)

        transpose = A.Compose([
            A.RandomCrop(width=128, height=128),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        image = transpose(image=image)["image"]
        label = torch.from_numpy(image.astype(np.float32).transpose([2, 0, 1]) / 255.)
        data = imresize(image, 1 / 4, 'bicubic')
        data = torch.from_numpy(data.astype(np.float32).transpose([2, 0, 1]) / 255.)
        return data, label

    def __len__(self):
        return self._length


def test():
    dataset = lmdbDataset()
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    for low_res , high_res in loader:
        # plt.imshow(low_res.cpu().numpy().squeeze(0).transpose([1,2,0]))
        # plt.show()
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
