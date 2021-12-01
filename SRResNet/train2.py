from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import model_utils
import utils
import os
import torch
from torch.backends import cudnn
from model import G
from torch import nn, optim
from SRResNetdatasets import SRResNetValDataset, DIV2KDataset, DIV2KSubDataset
from torch.utils.data.dataloader import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train_model(config, from_pth=False):
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
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=config['num_workers'],
                                  batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataset = SRResNetValDataset(config['val_file'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    print("===> Building model")
    model = G()
    if not from_pth:
        model.init_weight()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # ----END------
    start_step, best_step, best_psnr, writer, csv_file = \
        model_utils.load_checkpoint_iter(config['weight_file'], model, optimizer, csv_file,
                                    from_pth, auto_lr=config['auto_lr'])

    if torch.cuda.device_count() > 1:
        print("Using GPUs.\n")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    tb_writer = {'scalar': SummaryWriter(f"{logs_dir}/scalar"),
                 'test': SummaryWriter(f"{logs_dir}/test")}
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    num_steps = config['num_steps']
    iter_of_epoch = 1000
    global_info = {'device': device, 'step': start_step, 't': None, 'auto_lr': config['auto_lr'], 'milestone': [2e5],
                   'tb_writer': tb_writer, 'outputs_dir': outputs_dir, 'csv_writer': writer, 'num_steps': num_steps,
                   'best_psnr': best_psnr, 'best_step': best_step, 'iter_of_epoch': iter_of_epoch}

    while global_info['step'] < config['num_steps']:
        with tqdm(total=len(train_dataset)) as t:
            t.set_description('step:{}/{}'.format(global_info['step'], num_steps - 1))
            global_info['t'] = t
            model_utils.train_iter(model, dataloaders, optimizer, criterion, global_info)
    print('best step: {}, psnr: {:.2f}'.format(global_info['best_step'], global_info['best_psnr']))
    csv_file.close()
