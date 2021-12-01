import os
import sys
from train import train_model
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]).split('.')[0]
    scale = 3
    label_size = 19
    config = {'train_file': f'../datasets/T91_aug_label={label_size}_train_FSRCNNx{scale}_res.h5',
              'val_file': f'../datasets/Set5_label={label_size}_val_FSRCNNx{scale}_res.h5',
              'outputs_dir': f'../weight_file/{program}/',
              'logs_dir': f'../logs/{program}/',
              'csv_name': f'{program}.csv',
              'weight_file': f'./weight_file/{program}/latest.pth',
              'scale': scale,
              'in_size': 11,
              'out_size': label_size,
              'd': 56,
              's': 12,
              'm': 4,
              'lr': 1e-1,
              'batch_size': 128,
              'num_epochs': 100000,
              'num_workers': 4,
              'seed': 123,
              'init': 'MSRA',
              'Gpu': '0',
              'auto_lr': False,
              'residual': True
              }

    train_model(config, from_pth=True)