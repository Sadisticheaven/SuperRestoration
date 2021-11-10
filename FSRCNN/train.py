from tqdm import tqdm
import model_utils
import utils
import os
import torch
from torch.backends import cudnn
from model import FSRCNN
from torch import nn, optim
from FSRCNNdatasets import TrainDataset, ValDataset, ResValDataset
from torch.utils.data.dataloader import DataLoader
# 导入Visdom类
from visdom import Visdom


def train_model(config, from_pth=False, useVisdom=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['Gpu']
    if useVisdom:
        viz = Visdom(env='FSRCNN')
    else:
        viz = None
    outputs_dir = config['outputs_dir']
    lr = config['lr']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    utils.mkdirs(outputs_dir)
    csv_file = outputs_dir + config['csv_name']

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])
    # ----需要修改部分------
    model = FSRCNN(config['scale'], config['in_size'], config['out_size'],
                   num_channels=1, d=config['d'], s=config['s'], m=config['m'])
    if not from_pth:
        model.init_weights(method=config['init'])
    criterion = nn.MSELoss().cuda()

    optimizer = optim.SGD([
        {'params': model.extract_layer.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.deconv_layer.parameters(), 'lr': lr * 0.1}
    ], lr=lr, momentum=0.9)  # 前两层学习率lr， 最后一层学习率lr*0.1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=50)
    train_dataset = TrainDataset(config['train_file'])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=True)
    if config['residual']:
        val_dataset = ResValDataset(config['val_file'])
    else:
        val_dataset = ValDataset(config['val_file'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)
    # ----END------

    start_epoch, best_epoch, best_psnr, writer, csv_file = \
        model_utils.load_checkpoint(config['weight_file'], model, optimizer, csv_file,
                                    from_pth, useVisdom, viz, config['auto_lr'])

    if torch.cuda.device_count() > 1:
        print("Using GPUs.\n")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    for epoch in range(start_epoch, num_epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'learning rate: {lr}\n')
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description(f'epoch:{epoch}/{num_epochs - 1}')
            epoch_losses = model_utils.train(model, train_dataloader, optimizer, criterion, device, t)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        epoch_psnr = model_utils.validate(model, val_dataloader, device, config['residual'])

        if config['auto_lr']:
            scheduler.step(epoch_psnr.avg)
        if useVisdom:
            utils.draw_line(viz, X=[best_epoch], Y=[epoch_losses.avg], win='Loss', linename='trainLoss')
            utils.draw_line(viz, X=[best_epoch], Y=[epoch_psnr.avg], win='PSNR', linename='valPSNR')

        best_epoch, best_psnr = model_utils.save_checkpoint(model, optimizer, epoch, epoch_losses,
                                                            epoch_psnr, best_psnr, best_epoch, outputs_dir, writer)
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))