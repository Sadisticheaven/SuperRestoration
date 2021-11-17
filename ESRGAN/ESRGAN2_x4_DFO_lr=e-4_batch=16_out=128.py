import os
import sys
from train_ESRGAN import train_model
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]).split('.')[0]
    scale = 4
    label_size = 128
    config = {'val_file': f'../datasets/Set5_label={label_size}_val_ESRGANx{scale}.h5',
              'outputs_dir': f'../weight_file/{program}/x{scale}/',
              'csv_name': f'{program}.csv',
              'weight_file': f'../weight_file/{program}/x{scale}/latest.pth',
              'logs_dir': f'../logs/{program}/',
              'scale': scale,
              'in_size': 32,
              'out_size': label_size,
              'gen_lr': 1e-4,
              'disc_lr': 1e-4,
              'batch_size': 16,
              'num_epochs': 10000,
              'num_workers': 32,
              'seed': 0,
              'Gpu': '0',
              'auto_lr': True,
              'gen_k': 1,
              'disc_k': 1,
              'adversarial_weight': 5e-3,
              'pixel_weight': 1e-2
              }

    train_model(config, pre_train=None, from_pth=False)
    # train_model(config, pre_train='./best.pth', from_pth=False)