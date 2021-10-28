import csv
from tqdm import tqdm
import utils
import os
import torch
from torch.backends import cudnn
from torchvision import models
from model import G, D
from torch import nn, optim
from SRResNetdatasets import SRResNetValDataset, SRResNetTrainDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
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
        viz = Visdom(env='SRResNet')
    train_file = config['train_file']
    val_file = config['val_file']
    outputs_dir = config['outputs_dir']
    scale = config['scale']
    in_size = config['in_size']
    out_size = config['out_size']
    lr = config['lr']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_workers = config['num_workers']
    seed = config['seed']
    weight_file = config['weight_file']
    utils.mkdirs(outputs_dir)
    csv_file = outputs_dir + config['csv_name']
    vgg_loss = config['vgg_loss']

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # ----需要修改部分------
    print("===> Loading datasets")
    train_dataset = SRResNetTrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=num_workers,
                                  batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataset = SRResNetValDataset(val_file)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)
    if vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict('./VGG19/vgg19-dcbb9e9d.pth')

        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model().cuda()

    print("===> Building model")
    model = G()
    if not from_pth:
        model.init_weight()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=50)
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
        # epoch_vgglosses = utils.AverageMeter()
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

                loss.backward()  # 反向传播
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
            # if vgg_loss:
            #     for data in train_dataloader:
            #         inputs, labels = data
            #         inputs = inputs.to(device)
            #         labels = labels.to(device)
            #
            #         optimizer.zero_grad()  # 每个iteration前清除梯度
            #         preds = model(inputs)
            #         loss = criterion(preds, labels)
            #
            #         netContent.zero_grad()
            #         content_input = netContent(preds)
            #         content_target = netContent(labels)
            #         content_target = content_target.detach()
            #         content_loss = criterion(content_input, content_target) * 1/12.75
            #         content_loss.backward(retain_graph=True)
            #         epoch_vgglosses.update(content_loss.item(), len(inputs))
            #
            #         loss.backward()  # 反向传播
            #         epoch_losses.update(loss.item(), len(inputs))
            #         optimizer.step()
            #
            #         t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            #         t.set_postfix(vggloss='{:.6f}'.format(epoch_vgglosses.avg))
            #         t.update(len(inputs))
            # else:
            #     for data in train_dataloader:
            #         inputs, labels = data
            #         inputs = inputs.to(device)
            #         labels = labels.to(device)
            #
            #         optimizer.zero_grad()  # 每个iteration前清除梯度
            #         preds = model(inputs)
            #         loss = criterion(preds, labels)
            #
            #         loss.backward()  # 反向传播
            #         epoch_losses.update(loss.item(), len(inputs))
            #         optimizer.step()
            #
            #         t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            #         t.update(len(inputs))

        if useVisdom:
            draw_line(viz, X=[best_epoch], Y=[epoch_losses.avg], win='Loss', linename='trainLoss')

        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        # torch.save(model.state_dict(), outputs_dir + f'epoch_{epoch}.pth')

        model.eval()
        epoch_psnr = utils.AverageMeter()
        for data in val_dataloader:
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)

            with torch.no_grad():
                preds = model(inputs)
            preds = preds.mul(255.0).cpu().numpy().squeeze(0)
            preds = preds.transpose([1, 2, 0])  #chw->hwc
            preds = np.clip(preds, 0.0, 255.0)
            preds = utils.rgb2ycbcr(preds).astype(np.float32)[..., 0]/255.
            epoch_psnr.update(utils.calc_psnr(preds, labels.numpy()[0, ...]/255.).item(), len(inputs))
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        if config['auto_lr']:
            scheduler.step(epoch_psnr.avg)
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