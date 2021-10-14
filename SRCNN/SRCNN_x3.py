import sys
from train import train_model
import os
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]).split('.')[0]
    scale = 3
    config = {'train_file': f'../datasets/T91_aug_train_SRCNNx{scale}.h5',
              # config = {'train_file': 'G:/Document/datasets/T91_aug_x3.h5',
              'val_file': f'../datasets/Set5_val_SRCNNx{scale}.h5',
              'outputs_dir': f'./weight_file/{program}/x{scale}/',
              'csv_name': f'{program}.csv',
              'scale': scale,
              'lr': 1e-2,
              'batch_size': 16,
              'num_epochs': 1000,
              'num_workers': 4,
              'seed': 123,
              'weight_file': f'./weight_file/{program}/x{scale}/latest.pth',
              'Gpu': '0'
              }
    train_model(config, from_pth=False, useVisdom=False)