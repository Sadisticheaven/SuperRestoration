from tqdm import tqdm
import model_utils
import utils
import os
import torch
from torch.backends import cudnn
from torch import nn, optim
from model import N2_10_4, CLoss, HuberLoss, FSRCNN
from FSRCNNdatasets import T91TrainDataset, T91ValDataset, T91ResValDataset
from torch.utils.data.dataloader import DataLoader
# 导入Visdom类
from visdom import Visdom


def train_model(config, from_pth=False, pre_train=None, useVisdom=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['Gpu']
    if useVisdom:
        viz = Visdom(env='FSRCNN')
    else:
        viz = None
    outputs_dir = config['outputs_dir']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    utils.mkdirs(outputs_dir)
    csv_file = outputs_dir + config['csv_name']
    lr = config['lr']

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])
    # ----需要修改部分------
    model = N2_10_4(config['scale'], config['in_size'], config['out_size'],
                    num_channels=1, d=config['d'], m=config['m'])
    # model = FSRCNN(scale, in_size, out_size, num_channels=1, d=config['d'], m=config['m'])
    if config['Loss'] == 'CLoss':
        criterion = CLoss(delta=config['delta']).cuda()
    elif config['Loss'] == 'Huber':
        criterion = HuberLoss(delta=config['delta']).cuda()
    else:
        criterion = nn.MSELoss().cuda()

    optimizer = optim.SGD([
        {'params': model.deconv_layer.parameters(), 'lr': lr},
    ], lr=lr, momentum=0.9)  # 前两层学习率lr， 最后一层学习率lr*0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    train_dataset = T91TrainDataset(config['train_file'])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=True)
    if config['residual']:
        val_dataset = T91ResValDataset(config['val_file'])
    else:
        val_dataset = T91ValDataset(config['val_file'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)
    # ----END------
    start_epoch, best_epoch, best_psnr, writer, csv_file = \
        model_utils.load_checkpoint(config['weight_file'], model, optimizer, csv_file, from_pth, useVisdom, viz, config['auto_lr'])

    if not from_pth:
        if not os.path.exists(pre_train):
            print(f'Weight file not exist!\n{pre_train}\n')
            raise "Error"
        checkpoint = torch.load(pre_train)
        model.load_state_dict(checkpoint['model'])
        model.deconv_layer.weight.data.normal_(mean=0.0, std=0.001)
        model.deconv_layer.bias.data.zero_()

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
                                                            epoch_psnr, best_psnr, outputs_dir, writer)
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))