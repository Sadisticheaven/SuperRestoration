import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


class T91TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(T91TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['data'][idx] / 255., 0), np.expand_dims(f['label'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class T91ValDataset(Dataset):
    def __init__(self, h5_file):
        super(T91ValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['data'][str(idx)][:, :] / 255., 0), np.expand_dims(f['label'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class MatlabTrainDataset(Dataset):
    def __init__(self, h5_file):
        super(MatlabTrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['data'][idx], f['label'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class MatlabValidDataset(Dataset):
    def __init__(self, h5_file):
        super(MatlabValidDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['data'][idx], f['label'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])