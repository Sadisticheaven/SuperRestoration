import csv
from tqdm import tqdm
import utils
import os
import torch
from torch.backends import cudnn
from torch import nn, optim
from model import N2_10_4, CLoss
from FSRCNNdatasets import T91TrainDataset, T91ValDataset, T91ResValDataset
from torch.utils.data.dataloader import DataLoader
# 导入Visdom类
from visdom import Visdom


def draw_line(viz, X, Y, win, linename):
    viz.line(Y=Y,
             X=X,
             win=win,
             update='append',
             name=linename)


def train_model(config, from_pth=False, useVisdom=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['Gpu']
    if useVisdom:
        viz = Visdom(env='FSRCNN')
    train_file = config['train_file']
    val_file = config['val_file']
    outputs_dir = config['outputs_dir']
    scale = config['scale']
    in_size = config['in_size']
    out_size = config['out_size']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_workers = config['num_workers']
    seed = config['seed']
    weight_file = config['weight_file']
    utils.mkdirs(outputs_dir)
    csv_file = outputs_dir + config['csv_name']

    lr = config['lr']
    gradient_clip = config['gradient_clip']
    weight_decay = config['weight_decay']

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    # ----需要修改部分------
    model = N2_10_4(scale, in_size, out_size, num_channels=1, d=config['d'], m=config['m'])
    if not from_pth:
        model.init_weights(method=config['init'])
    criterion = CLoss().cuda()

    extract_weight = [n.weight for n in model.extract_layer if isinstance(n, nn.Conv2d)]
    extract_PReLU = [n.weight for n in model.extract_layer if isinstance(n, nn.PReLU)]
    extract_bias = [n.bias for n in model.extract_layer if isinstance(n, nn.Conv2d)]
    mid_weight = [n.weight for n in model.mid_part if isinstance(n, nn.Conv2d)]
    mid_PReLU = [n.weight for n in model.mid_part if isinstance(n, nn.PReLU)]
    mid_bias = [n.bias for n in model.mid_part if isinstance(n, nn.Conv2d)]
    deconv_weight = model.deconv_layer.weight
    deconv_bias = model.deconv_layer.bias
    optimizer = optim.SGD([
        {'params': extract_weight, 'weight_decay': weight_decay},
        {'params': extract_PReLU},
        {'params': extract_bias, 'lr': lr},

        {'params': mid_weight, 'weight_decay': weight_decay},
        {'params': mid_PReLU},
        {'params': mid_bias, 'lr': lr},

        {'params': deconv_weight, 'weight_decay': weight_decay, 'lr':  lr * 0.1},
        {'params': deconv_bias, 'lr': lr * 0.1},
    # optimizer = optim.SGD([
    #     {'params': model.extract_layer.parameters()},
    #     {'params': model.mid_part.parameters()},
    #     {'params': model.deconv_layer.parameters(), 'lr': lr * 0.1},
    ], lr=lr, momentum=0.9)  # 前两层学习率lr， 最后一层学习率lr*0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    train_dataset = T91TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    if config['residual']:
        val_dataset = T91ResValDataset(val_file)
    else:
        val_dataset = T91ValDataset(val_file)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)
    # ----END------

    if from_pth:
        if not os.path.exists(weight_file):
            print(f'Weight file not exist!\n{weight_file}\n')
            raise "Error"
        checkpoint = torch.load(weight_file)
        csv_file = open(csv_file, 'a', newline='')
        writer = csv.writer(csv_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if not config['auto_lr']:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[2]['lr'] = lr * 0.1
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        best_epoch = checkpoint['epoch']
        start_epoch = best_epoch + 1
        best_psnr = checkpoint['psnr']
        print('Start from loss: {:.6f}, psnr: {:.2f}\n'.format(checkpoint['loss'], best_psnr))
        if useVisdom:
            draw_line(viz, X=[best_epoch], Y=[checkpoint['loss']], win='Loss', linename='trainLoss')
            draw_line(viz, X=[best_epoch], Y=[best_psnr], win='PSNR', linename='valPSNR')
    else:
        csv_file = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(('epoch', 'loss', 'psnr'))
        start_epoch = 0
        best_epoch = 0
        best_psnr = 0.0

    if torch.cuda.device_count() > 1:
        print("Using GPUs.\n")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = utils.AverageMeter()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'learning rate: {lr}\n')
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description(f'epoch:{epoch}/{num_epochs - 1}')

            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # 每个iteration前清除梯度
                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip)
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        if useVisdom:
            draw_line(viz, X=[best_epoch], Y=[epoch_losses.avg], win='Loss', linename='trainLoss')

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.eval()
        epoch_psnr = utils.AverageMeter()
        for data in val_dataloader:
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            if config['residual']:
                bicubic = data[2]
                bicubic = bicubic.to(device)

            with torch.no_grad():
                preds = model(inputs)
            if config['residual']:
                preds = preds + bicubic
            preds = preds.clamp(0.0, 1.0)
            epoch_psnr.update(utils.calc_psnr(preds, labels).item(), len(inputs))
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        if config['auto_lr']:
            scheduler.step()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'loss': epoch_losses.avg, 'psnr': epoch_psnr.avg}
        torch.save(state, outputs_dir + f'latest.pth')
        writer.writerow((state['epoch'], state['loss'], state['psnr']))
        if useVisdom:
            draw_line(viz, X=[best_epoch], Y=[epoch_psnr.avg], win='PSNR', linename='valPSNR')

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            torch.save(state, outputs_dir + 'best.pth')
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))