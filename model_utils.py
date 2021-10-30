import csv
import os
import torch
from tqdm import tqdm

import utils


def load_checkpoint(weight_file, model, optimizer, csv_file, from_pth=False, useVisdom=False, viz=None, auto_lr=False):
    if from_pth:
        if not os.path.exists(weight_file):
            print(f'Weight file not exist!\n{weight_file}\n')
            raise "Error"
        checkpoint = torch.load(weight_file)
        csv_file = open(csv_file, 'a', newline='')
        writer = csv.writer(csv_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        old_param = optimizer.param_groups
        optimizer.load_state_dict(checkpoint['optimizer'])
        if not auto_lr:
            optimizer.param_groups = old_param
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        best_epoch = checkpoint['epoch']
        start_epoch = best_epoch + 1
        best_psnr = checkpoint['psnr']
        print('Start from loss: {:.6f}, psnr: {:.2f}\n'.format(checkpoint['loss'], best_psnr))
        if useVisdom:
            utils.draw_line(viz, X=[best_epoch], Y=[checkpoint['loss']], win='Loss', linename='trainLoss')
            utils.draw_line(viz, X=[best_epoch], Y=[best_psnr], win='PSNR', linename='valPSNR')
    else:
        csv_file = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(('epoch', 'loss', 'psnr'))
        start_epoch = 0
        best_epoch = 0
        best_psnr = 0.0
    return start_epoch, best_epoch, best_psnr, writer, csv_file


def train(model, train_dataloader, optimizer, criterion, device, t):
    model.train()
    epoch_losses = utils.AverageMeter()
    for data in train_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 每个iteration前清除梯度
        preds = model(inputs)
        loss = criterion(preds, labels)
        epoch_losses.update(loss.item(), len(inputs))

        loss.backward()  # 反向传播
        optimizer.step()

        t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
        t.update(len(inputs))
    return epoch_losses


def validate(model, val_dataloader, device, residual=False):
    model.eval()
    epoch_psnr = utils.AverageMeter()
    for data in val_dataloader:
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        if residual:
            bicubic = data[2]
            bicubic = bicubic.to(device)
        with torch.no_grad():
            preds = model(inputs)
        if residual:
            preds = preds + bicubic
        preds = preds.clamp(0.0, 1.0)
        epoch_psnr.update(utils.calc_psnr(preds, labels).item(), len(inputs))
    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
    return epoch_psnr


def save_checkpoint(model, optimizer, epoch, epoch_losses, epoch_psnr, best_psnr, best_epoch, outputs_dir, csv_writer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
             'loss': epoch_losses.avg, 'psnr': epoch_psnr.avg}
    torch.save(state, outputs_dir + f'latest.pth')
    csv_writer.writerow((state['epoch'], state['loss'], state['psnr']))

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        torch.save(state, outputs_dir + 'best.pth')
    return best_epoch, best_psnr
