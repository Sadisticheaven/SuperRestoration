import os
import sys
from train_SRGAN import train_model
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]).split('.')[0]
    scale = 4
    label_size = 96
    config = {'train_file': f'../datasets/test/',
              'val_file': f'../datasets/Set5_label={label_size}_val_SRResNetx{scale}.h5',
              'outputs_dir': f'./weight_file/{program}/x{scale}/',
              'csv_name': f'{program}.csv',
              'weight_file': f'./weight_file/{program}/x{scale}/latest.pth',
              'logs_dir': f'./logs/{program}/',
              'scale': scale,
              'in_size': 24,
              'out_size': label_size,
              'gen_lr': 1e-4,
              'disc_lr': 1e-4,
              'batch_size': 5,
              'num_epochs': 10000,
              'num_workers': 0,
              'seed': 123,
              'init': 'MSRA',
              'Gpu': '0',
              'gen_k': 1,
              'disc_k': 1,
              }

    train_model(config, pre_train=None, from_pth=False)
    # train_model(config, pre_train='./best.pth', from_pth=False, use_visdom=False)