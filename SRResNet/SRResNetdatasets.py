import h5py
import numpy as np
from torch.utils.data import Dataset


class SRResNetTrainDataset(Dataset):
    def __init__(self, h5_file):
        super(SRResNetTrainDataset, self).__init__()
        hf = h5py.File(h5_file)
        self.data = hf.get("data")
        self.target = hf.get("label")

    def __getitem__(self, idx):
        return np.expand_dims(self.data[idx] / 255., 0), np.expand_dims(self.target[idx] / 255. * 2. - 1, 0)

    def __len__(self):
        return self.data.shape[0]


class SRResNetValDataset(Dataset):
    def __init__(self, h5_file):
        super(SRResNetValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['data'][str(idx)][:, :] / 255., 0), np.expand_dims(f['label'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])
