import os
import sys
from train import train_model
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]).split('.')[0]
    scale = 4
    label_size = 96
    config = {'train_file': f'../datasets/DIV2K_train_HR/',
              'val_file': f'../datasets/Set5_label={label_size}_val_SRResNetx{scale}.h5',
              'outputs_dir': f'./weight_file/{program}/x{scale}/',
              'logs_dir': f'./logs/{program}/',
              'csv_name': f'{program}.csv',
              'weight_file': f'./weight_file/{program}/x{scale}/latest.pth',
              'scale': scale,
              'in_size': 11,
              'out_size': label_size,
              'lr': 1e-4,
              'batch_size': 16,
              'num_epochs': 10000,
              'num_workers': 4,
              'seed': 123,
              'init': 'MSRA',
              'Gpu': '0',
              'auto_lr': False
              }

    train_model(config, from_pth=False, useVisdom=False)