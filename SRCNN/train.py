import csv

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
import os
import torch
from torch.backends import cudnn
from models import SRCNN
from torch import nn, optim
from SRCNNdatasets import TrainDataset, ValDataset
from torch.utils.data.dataloader import DataLoader
import model_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train_model(config, from_pth=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['Gpu']

    outputs_dir = config['outputs_dir']
    lr = config['lr']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    csv_file = outputs_dir + config['csv_name']
    logs_dir = config['logs_dir']
    utils.mkdirs(outputs_dir)
    utils.mkdirs(logs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])

    model = SRCNN()
    if not from_pth:
        model.init_weights()
    criterion = nn.MSELoss()
    optimizer = optim.SGD([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr * 0.1}
    ], lr=lr, momentum=0.9)  # 前两层学习率lr， 最后一层学习率lr*0.1

    train_dataset = TrainDataset(config['train_file'])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  drop_last=True)
    val_dataset = ValDataset(config['val_file'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    start_epoch, best_epoch, best_psnr, writer, csv_file = \
        model_utils.load_checkpoint(config['weight_file'], model, optimizer, csv_file, from_pth)

    if torch.cuda.device_count() > 1:
        print("Using GPUs.\n")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    writer_scalar = SummaryWriter(f"{logs_dir}/scalar")

    for epoch in range(start_epoch, num_epochs):
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description(f'epoch:{epoch}/{num_epochs - 1}')
            epoch_losses = model_utils.train(model, train_dataloader, optimizer, criterion, device, t)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        epoch_psnr = model_utils.validate(model, val_dataloader, device)

        writer_scalar.add_scalar('Loss', epoch_losses.avg, epoch)
        writer_scalar.add_scalar('PSNR', epoch_psnr.avg, epoch)

        best_epoch, best_psnr = model_utils.save_checkpoint(model, optimizer, epoch, epoch_losses,
                                                            epoch_psnr, best_psnr, best_epoch, outputs_dir, writer)

    csv_file.close()
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))









