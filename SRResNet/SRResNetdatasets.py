import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

class SRResNetTrainDataset(Dataset):
    def __init__(self, h5_file):
        super(SRResNetTrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['data'][idx] / 255.), torch.from_numpy(f['label'][idx] / 255. * 2. - 1)
            # return torch.from_numpy(f['data'][idx] / 255.).permute(2, 0, 1), torch.from_numpy(f['label'][idx] / 255. * 2. - 1).permute(2, 0, 1)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class SRResNetValDataset(Dataset):
    def __init__(self, h5_file):
        super(SRResNetValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return torch.from_numpy(f['data'][str(idx)][:, :, :] / 255.), torch.from_numpy(f['label'][str(idx)][:, :, :] / 255.)
            # return torch.from_numpy(f['data'][str(idx)][:, :] / 255.).permute(2, 0, 1), torch.from_numpy(f['label'][str(idx)][:, :] / 255.).permute(2, 0, 1)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])
