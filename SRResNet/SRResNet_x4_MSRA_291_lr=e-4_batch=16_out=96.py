import os
import sys
from train import train_model
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]).split('.')[0]
    scale = 4
    label_size = 96
    config = {'train_file': f'../datasets/291_aug_label={label_size}_train_SRResNetx{scale}.h5',
              'val_file': f'../datasets/Set5_label={label_size}_val_SRResNetx{scale}.h5',
              'outputs_dir': f'./weight_file/{program}/x{scale}/',
              'csv_name': f'{program}.csv',
              'weight_file': f'./weight_file/{program}/x{scale}/latest.pth',
              'scale': scale,
              'in_size': 11,
              'out_size': label_size,
              'lr': 1e-4,
              'batch_size': 16,
              'num_epochs': 100000,
              'num_workers': 2,
              'seed': 123,
              'init': 'MSRA',
              'Gpu': '0',
              'auto_lr': False,
              'vgg_loss': False
              }

    train_model(config, from_pth=False, useVisdom=False)