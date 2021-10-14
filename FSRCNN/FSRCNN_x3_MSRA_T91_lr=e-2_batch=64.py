import os
import sys
from train import train_model
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]).split('.')[0]
    scale = 3
    config = {'train_file': f'../datasets/T91_aug_train_FSRCNNx{scale}.h5',
              'val_file': f'../datasets/Set5_val_FSRCNNx{scale}.h5',
              'outputs_dir': f'./weight_file/{program}/x{scale}/',
              'csv_name': f'{program}.csv',
              'weight_file': f'./weight_file/{program}/x{scale}/latest.pth',
              'scale': scale,
              'in_size': 11,
              'out_size': 19,
              'd': 56,
              's': 12,
              'm': 4,
              'lr': 1e-2,
              'batch_size': 64,
              'num_epochs': 100000,
              'num_workers': 4,
              'seed': 123,
              'init': 'MSRA',
              'Gpu': '0',
              'auto_lr': False
              }

    train_model(config, from_pth=False, useVisdom=False)