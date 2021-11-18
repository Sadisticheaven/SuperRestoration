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
        load_optimizer(optimizer, checkpoint['optimizer'], auto_lr)

        best_epoch = checkpoint['epoch']
        start_epoch = best_epoch + 1
        best_psnr = checkpoint['psnr']
        print('Start from loss: {:.6f}, psnr: {:.2f}\n'.format(checkpoint['loss'], best_psnr))
    else:
        csv_file = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(('epoch', 'loss', 'psnr'))
        start_epoch = 0
        best_epoch = 0
        best_psnr = 0.0
    return start_epoch, best_epoch, best_psnr, writer, csv_file


def load_GAN_checkpoint(pre_train, weight_file, gen, gen_opt, disc, disc_opt, csv_file, from_pth=False, auto_lr=False):
    if from_pth:
        if not os.path.exists(weight_file):
            print(f'Weight file not exist!\n{weight_file}\n')
            raise "Error"
        checkpoint = torch.load(weight_file)
        csv_file = open(csv_file, 'a', newline='')
        writer = csv.writer(csv_file)

        gen.load_state_dict(checkpoint['gen'])
        load_optimizer(gen_opt, checkpoint['gen_opt'], auto_lr)

        disc.load_state_dict(checkpoint['disc'])
        load_optimizer(disc_opt, checkpoint['disc_opt'], auto_lr)

        best_epoch = checkpoint['epoch']
        start_epoch = best_epoch + 1
        best_psnr = checkpoint['psnr']
        best_niqe = checkpoint['niqe']
        print('Start from Gloss: {:.6f}, psnr: {:.2f}, Dloss: {:.6f}, niqe: {:.4f}\n'.format(
            checkpoint['Gloss'], best_psnr, checkpoint['Dloss'], best_niqe))
    else:
        csv_file = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(('epoch', 'Gloss', 'psnr', 'Dloss', 'niqe'))
        if pre_train is not None:
            if not os.path.exists(pre_train):
                print(f'Weight file not exist!\n{pre_train}\n')
                raise "Error"
            checkpoint = torch.load(pre_train)
            gen.load_state_dict(checkpoint['model'])
        else:
            gen.init_weight()
        start_epoch = 0
        best_epoch = 0
        best_psnr = 0.0
        best_niqe = 100.0
    return start_epoch, best_epoch, best_psnr, best_niqe, writer, csv_file


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


def save_GAN_checkpoint(gen, gen_opt, disc, disc_opt, epoch, G_losses, D_losses, epoch_psnr, epoch_niqe, best_psnr, best_niqe, best_epoch, outputs_dir, csv_writer):
    state = {'gen': gen.state_dict(), 'gen_opt': gen_opt.state_dict(),
             'disc': disc.state_dict(), 'disc_opt': disc_opt.state_dict(),
             'epoch': epoch, 'Gloss': G_losses.avg, 'Dloss': D_losses.avg,
             'psnr': epoch_psnr.avg, 'niqe': epoch_niqe.avg}
    torch.save(state, outputs_dir + f'latest.pth')
    csv_writer.writerow((state['epoch'], state['Gloss'], state['psnr'], state['Dloss'], state['niqe']))

    if epoch_niqe.avg < best_niqe:
        best_epoch = epoch
        best_niqe = epoch_niqe.avg
        torch.save(state, outputs_dir + f'best_niqe.pth')
    if epoch_psnr.avg > best_psnr:
        best_psnr = epoch_psnr.avg
        torch.save(state, outputs_dir + 'best_psnr.pth')
    return best_epoch, best_psnr, best_niqe


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_lr(optimizer, gamma):
    for param_group in optimizer.param_groups:
         param_group['lr'] = param_group['lr'] * gamma


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
         param_group['lr'] = lr


def load_optimizer(opt, file_path, auto_lr=False):
    base_lr = opt.param_groups[0]['lr']
    opt.load_state_dict(file_path)
    if not auto_lr:
        set_lr(opt, base_lr)
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()