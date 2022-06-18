import torch
from torch.backends import cudnn
from model import G, D
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    config = {'weight_file': './weight_file/SRGAN7_x4_DIVsub_WGAN/',
              'scale': 4
              }
    scale = config['scale']
    padding = scale
    # weight_file = config['weight_file'] + f'best.pth'
    weight_file = config['weight_file'] + f'x{scale}/latest.pth'
    if not os.path.exists(weight_file):
        print(f'Weight file not exist!\n{weight_file}\n')
        raise "Error"

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(weight_file)

    disc = D().to(device)
    disc = torch.nn.DataParallel(disc)
    disc.load_state_dict(checkpoint['disc'])
    disc = disc.module
    checkpoint['disc'] = disc.state_dict()
    torch.save(checkpoint, weight_file)
