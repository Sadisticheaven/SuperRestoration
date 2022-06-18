
from tqdm import tqdm
import model_utils
import utils
import os
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from VGGLoss import VGGLoss
from model import G, D
from torch import nn, optim
from SRResNetdatasets import SRResNetValDataset, SRResNetTrainDataset, DIV2KDataset, DIV2KSubDataset
from torch.utils.data.dataloader import DataLoader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train_model(config, pre_train, from_pth=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['Gpu']
    outputs_dir = config['outputs_dir']
    batch_size = config['batch_size']
    utils.mkdirs(outputs_dir)
    csv_file = outputs_dir + config['csv_name']
    logs_dir = config['logs_dir']
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])

    # ----需要修改部分------
    print("===> Loading datasets")
    train_dataset = DIV2KSubDataset()
    # train_dataset = DIV2KDataset(config['train_file'])
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=config['num_workers'],
                                  batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataset = SRResNetValDataset(config['val_file'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    print("===> Building model")
    gen = G()
    disc = D()
    if not from_pth:
        disc.init_weight()
    criterion = {'pixel_loss': nn.MSELoss().to(device), 'vgg_loss': VGGLoss(device)}
    gen_opt = optim.RMSprop(gen.parameters(), lr=config['gen_lr'])
    disc_opt = optim.RMSprop(disc.parameters(), lr=config['disc_lr'])
    # ----END------
    start_step, best_step, best_niqe, writer, csv_file = \
        model_utils.load_GAN_checkpoint_iter(pre_train, config['weight_file'], gen, gen_opt, disc, disc_opt,
                                             csv_file, from_pth, config['auto_lr'])

    if torch.cuda.device_count() > 1:
        print("Using GPUs.\n")
        gen = torch.nn.DataParallel(gen)
        disc = torch.nn.DataParallel(disc)
    gen = gen.to(device)
    disc = disc.to(device)

    tb_writer = {'scalar': SummaryWriter(f"{logs_dir}/scalar"), 'test': SummaryWriter(f"{logs_dir}/test")}
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    num_steps = config['num_steps']
    iter_of_epoch = 1000
    global_info = {'device': device, 'step': start_step, 't': None, 'auto_lr': config['auto_lr'],
                   'milestone': config['milestone'],
                   'tb_writer': tb_writer, 'outputs_dir': outputs_dir, 'csv_writer': writer, 'num_steps': num_steps,
                   'best_step': best_step, 'batch_no': -1, 'iter_of_epoch': iter_of_epoch, 'best_niqe': best_niqe,
                   'pixel_weight': config['pixel_weight'], 'adversarial_weight': config['adversarial_weight'],
                   'disc_k': config['gen_k'], 'gen_k': config['disc_k']}
    log_dict = {'D_losses': 0., 'G_losses': 0.,
                'D_losses_real': 0., 'D_losses_fake': 0.,
                'pixel_loss': 0., 'gan_loss': 0.,
                'percep_loss': 0.,
                'F_prob': 0., 'R_prob': 0.}
    while global_info['step'] < config['num_steps']:
        with tqdm(total=len(train_dataset)) as t:
            t.set_description('step:{}/{}'.format(global_info['step'], num_steps - 1))
            global_info['t'] = t
            model_utils.train_SRGAN_iter(gen, disc, dataloaders, gen_opt, disc_opt, criterion, global_info, log_dict, use_WGAN=True)

    print('best step: {}, niqe: {:.2f}'.format(global_info['best_step'], global_info['best_niqe']))
