import os
from PIL import Image
import numpy as np
import albumentations as A
from imresize import imresize
import h5py
from torch.utils.data import Dataset, DataLoader
import torch


class SRResNetTrainDataset(Dataset):
    def __init__(self, h5_file):
        super(SRResNetTrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['data'][idx] / 255.), torch.from_numpy(f['label'][idx] / 255. * 2 - 1)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class SRResNetValDataset(Dataset):
    def __init__(self, h5_file):
        super(SRResNetValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['data'][str(idx)][:, :, :] / 255.), \
                   torch.from_numpy(f['label'][str(idx)][:, :, :] / 255.)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class DIV2KDataset(Dataset):
    def __init__(self, root_dir='../datasets/DIV2K_train_HR/'):
        super(DIV2KDataset, self).__init__()
        self.data = []
        self.img_names = os.listdir(root_dir)

        for name in self.img_names:
            self.data.append(root_dir + name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]

        image = np.array(Image.open(img_path))
        transpose = A.RandomCrop(width=96, height=96)
        image = transpose(image=image)["image"]
        # transpose2 = A.Normalize(0.5, 0.5)
        # label = transpose2(image=image)["image"]
        # label = torch.from_numpy(label.astype(np.float32).transpose([2, 0, 1]))
        label = torch.from_numpy(image.astype(np.float32).transpose([2, 0, 1])/127.5-1)
        data = imresize(image, 1 / 4, 'bicubic')
        data = torch.from_numpy(data.astype(np.float32).transpose([2, 0, 1]) / 255)
        return data, label


def test():
    dataset = DIV2KDataset(root_dir='../datasets/T91/')
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()