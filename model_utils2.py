import copy
import csv
import os
import numpy as np
import torch
import torchvision

import niqe
import utils


def load_checkpoint(weight_file, model, optimizer, csv_file, from_pth=False, auto_lr=False):
    if from_pth:
        if not os.path.exists(weight_file):
            print(f'Weight file not exist!\n{weight_file}\n')
            raise "Error"
        checkpoint = torch.load(weight_file)
        csv_file = open(csv_file, 'a', newline='')
        writer = csv.writer(csv_file)
        model.load_state_dict(checkpoint['model'])
        load_optimizer(optimizer, checkpoint['optimizer'], auto_lr)

        best_epoch = checkpoint['best_epoch']
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']
        print('Start from loss: {:.6f}, psnr: {:.2f}\n'.format(checkpoint['loss'], checkpoint['psnr']))
    else:
        csv_file = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(('epoch', 'loss', 'psnr'))
        start_epoch = 0
        best_epoch = 0
        best_psnr = 0.0
    return start_epoch, best_epoch, best_psnr, writer, csv_file


def load_checkpoint_iter(weight_file, model, optimizer, csv_file, from_pth=False, auto_lr=False):
    if from_pth:
        if not os.path.exists(weight_file):
            print(f'Weight file not exist!\n{weight_file}\n')
            raise "Error"
        checkpoint = torch.load(weight_file)
        csv_file = open(csv_file, 'a', newline='')
        writer = csv.writer(csv_file)
        model.load_state_dict(checkpoint['model'])
        load_optimizer(optimizer, checkpoint['optimizer'], auto_lr)

        best_step = checkpoint['best_step']
        start_step = checkpoint['step'] + 1
        best_psnr = checkpoint['best_psnr']
        print('Start from loss: {:.6f}, psnr: {:.2f}\n'.format(checkpoint['loss'], checkpoint['psnr']))
    else:
        csv_file = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(('step', 'loss', 'psnr'))
        start_step = 0
        best_step = 0
        best_psnr = 0.0
    return start_step, best_step, best_psnr, writer, csv_file


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

        best_epoch = checkpoint['best_epoch']
        start_epoch = checkpoint['epoch'] + 1
        best_niqe = checkpoint['best_niqe']
        print('Start from Gloss: {:.6f}, psnr: {:.2f}, Dloss: {:.6f}, niqe: {:.4f}\n'.format(
            checkpoint['Gloss'], checkpoint['psnr'], checkpoint['Dloss'], checkpoint['niqe']))
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
        best_niqe = 100.0
    return start_epoch, best_epoch, best_niqe, writer, csv_file


def load_GAN_checkpoint_iter(pre_train, weight_file, gen, gen_opt, disc, disc_opt, csv_file, from_pth=False, auto_lr=False):
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

        best_step = checkpoint['best_step']
        start_step = checkpoint['step'] + 1
        best_niqe = checkpoint['best_niqe']
        print('Start from Gloss: {:.6f}, psnr: {:.2f}, Dloss: {:.6f}, niqe: {:.4f}\n'.format(
            checkpoint['Gloss'], checkpoint['psnr'], checkpoint['Dloss'], checkpoint['niqe']))
    else:
        csv_file = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(('step', 'Gloss', 'psnr', 'Dloss', 'niqe'))
        if pre_train is not None:
            if not os.path.exists(pre_train):
                print(f'Weight file not exist!\n{pre_train}\n')
                raise "Error"
            checkpoint = torch.load(pre_train)
            gen.load_state_dict(checkpoint['model'])
        else:
            gen.init_weight()
        start_step = 0
        best_step = 0
        best_niqe = 100.0
    return start_step, best_step, best_niqe, writer, csv_file


def validate(model, val_dataloader, device, residual=False):
    model.eval()
    epoch_psnr = utils.AverageMeter()
    for idx, data in enumerate(val_dataloader):
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


def val_SRResNet(model, val_dataloader, global_info):
    model.eval()
    epoch_psnr = utils.AverageMeter()
    for idx, data in enumerate(val_dataloader):
        inputs, labels = data
        inputs = inputs.to(global_info['device'])
        with torch.no_grad():
            preds = model(inputs) * 0.5 + 0.5
            img_grid_fake = torchvision.utils.make_grid(preds, normalize=True)
            global_info['tb_writer']['test'].add_image(f"Test Fake{idx}", img_grid_fake, global_step=global_info['step'])
        preds = preds.mul(255.0).cpu().numpy().squeeze(0)
        preds = preds.transpose([1, 2, 0])  # chw->hwc
        preds = np.clip(preds, 0.0, 255.0)
        preds = utils.rgb2ycbcr(preds).astype(np.float32)[..., 0] / 255.
        epoch_psnr.update(utils.calc_psnr(preds, labels.numpy()[0, 0, ...]).item(), len(inputs))
    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
    return epoch_psnr


def val_SRGAN(gen, val_dataloader, global_info):
    gen.eval()
    epoch_psnr = utils.AverageMeter()
    epoch_niqe = utils.AverageMeter()
    for idx, data in enumerate(val_dataloader):
        inputs, labels = data
        inputs = inputs.to(global_info['device'])
        with torch.no_grad():
            fake = gen(inputs) * 0.5 + 0.5
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            global_info['tb_writer']['test'].add_image(f"Test Fake{idx}", img_grid_fake,
                                                       global_step=global_info['step'])
        fake = fake.mul(255.0).cpu().numpy().squeeze(0)
        fake = fake.transpose([1, 2, 0])  # chw->hwc
        fake = np.clip(fake, 0.0, 255.0)
        fake = utils.rgb2ycbcr(fake).astype(np.float32)[..., 0] / 255.
        epoch_niqe.update(niqe.calculate_niqe(fake), len(inputs))
        epoch_psnr.update(utils.calc_psnr(fake, labels.numpy()[0, 0, ...]).item(), len(inputs))
    print('eval psnr: {:.2f}, niqe: {:.4f}'.format(epoch_psnr.avg, epoch_niqe.avg))
    global_info['tb_writer']['scalar'].add_scalar('PSNR', epoch_psnr.avg, global_info['step'])
    global_info['tb_writer']['scalar'].add_scalar('NIQE', epoch_niqe.avg, global_info['step'])
    global_info.update({'psnr': epoch_psnr.avg, 'niqe': epoch_niqe.avg})


def val_ESRGAN(gen, val_dataloader, global_info):
    gen.eval()
    epoch_psnr = utils.AverageMeter()
    epoch_niqe = utils.AverageMeter()
    for idx, data in enumerate(val_dataloader):
        inputs, labels = data
        inputs = inputs.to(global_info['device'])
        with torch.no_grad():
            fake = gen(inputs)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            global_info['tb_writer']['test'].add_image(f"Test Fake{idx}", img_grid_fake,
                                                       global_step=global_info['step'])
        fake = fake.mul(255.0).cpu().numpy().squeeze(0)
        fake = fake.transpose([1, 2, 0])  # chw->hwc
        fake = np.clip(fake, 0.0, 255.0)
        fake = utils.rgb2ycbcr(fake).astype(np.float32)[..., 0] / 255.
        epoch_niqe.update(niqe.calculate_niqe(fake), len(inputs))
        epoch_psnr.update(utils.calc_psnr(fake, labels.numpy()[0, 0, ...]).item(), len(inputs))
    print('eval psnr: {:.2f}, niqe: {:.4f}'.format(epoch_psnr.avg, epoch_niqe.avg))
    global_info['tb_writer']['scalar'].add_scalar('PSNR', epoch_psnr.avg, global_info['step'])
    global_info['tb_writer']['scalar'].add_scalar('NIQE', epoch_niqe.avg, global_info['step'])
    global_info.update({'psnr': epoch_psnr.avg, 'niqe': epoch_niqe.avg})


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


def train_iter(model, dataloaders, optimizer, criterion, global_info):
    model.train()
    epoch_losses = utils.AverageMeter()
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    device = global_info['device']
    for batch_idx, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 每个iteration前清除梯度
        preds = model(inputs)
        loss = criterion(preds, labels)
        epoch_losses.update(loss.item(), len(inputs))

        loss.backward()  # 反向传播
        optimizer.step()
        # update bar
        global_info['t'].set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
        global_info['t'].update(len(inputs))
        # log per epoch
        if (batch_idx + 1) % global_info['iter_of_epoch'] == 0:
            global_info['step'] += 1
            step = global_info['step']
            # update learning rate
            if global_info['auto_lr'] and step in global_info['milestone']:
                update_lr(optimizer, 0.1)
                print(f'learning rate: {get_lr(optimizer)}\n')
            # for multi-GPU
            if isinstance(model, torch.nn.DataParallel):
                model2 = model.module
            else:
                model2 = copy.deepcopy(model)
            # validation and log
            model2.eval()
            epoch_psnr = val_SRResNet(model2, val_dataloader, global_info)
            global_info['tb_writer']['scalar'].add_scalar('PSNR', epoch_psnr.avg, step)
            global_info['tb_writer']['scalar'].add_scalar('Loss', epoch_losses.avg, step)
            # save
            save_checkpoint_iter(model2, optimizer, global_info, epoch_losses, epoch_psnr)
            # update for next iteration
            epoch_losses = utils.AverageMeter()
            global_info['t'].set_description('step:{}/{}'.format(step, global_info['num_steps'] - 1), refresh=True)


def iterate_SRWGAN(gen, disc, gen_opt, disc_opt, criterion, batch_idx, data, global_info, log_dict):
    device = global_info['device']
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    # Train Disciminator  max(ED(x) - ED(G(z)))
    fake = gen(inputs)
    if batch_idx % global_info['disc_k'] == 0:
        disc_real = disc(labels)
        disc_fake = disc(fake.detach())
        disc_loss = -(disc_real.mean() - disc_fake.mean())
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()
        log_dict['D_losses'].update(disc_loss.item(), len(inputs))
        for p in disc.parameters():
            p.data.clamp_(-0.01, 0.01)
    else:
        with torch.no_grad():
            disc_real = disc(labels)
            disc_fake = disc(fake.detach())
    # Train Generator min(-ED(G(z))) <-> max ED(G(z))
    if batch_idx % global_info['gen_k'] == 0:
        disc_fake = disc(fake)
        adversarial_loss = -disc_fake.mean()
        content_loss = 2e-6 * criterion['vgg_loss']((fake+1.)/2, (labels+1.)/2)
        # content_loss = 0.006 * criterion['vgg_loss']((fake+1.)/2, (labels+1.)/2)
        gen_loss = content_loss + global_info['adversarial_weight'] * adversarial_loss + \
                   global_info['pixel_weight'] * criterion['pixel_loss'](fake, labels)
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        log_dict['G_losses'].update(gen_loss.item(), len(inputs))
    # log
    log_dict['F_prob'].update(disc_fake.mean(), 1)
    log_dict['R_prob'].update(disc_real.mean(), 1)
    global_info['t'].set_postfix(loss='Gloss: {:.6f}, Dloss: {:.6f}, fake: {:.2f}, real: {:.2f}'
                                 .format(log_dict['G_losses'].avg, log_dict['D_losses'].avg,
                                         log_dict['F_prob'].avg, log_dict['R_prob'].avg))
    global_info['t'].update(len(inputs))


def iterate_SRGAN(gen, disc, gen_opt, disc_opt, criterion, batch_idx, data, global_info, log_dict):
    device = global_info['device']
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    # Train Disciminator  max(logD(x) + log(1-D(G(z))))
    fake = gen(inputs)
    if batch_idx % global_info['disc_k'] == 0:
        disc_real = disc(labels)
        disc_fake = disc(fake.detach())
        disc_loss_real = criterion['bce'](disc_real, torch.ones_like(disc_real))
        disc_loss_fake = criterion['bce'](disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()
        log_dict['D_losses'].update(disc_loss.item(), len(inputs))
    else:
        with torch.no_grad():
            disc_real = disc(labels)
            disc_fake = disc(fake.detach())
    # Train Generator min(log(1-D(G(z))) <-> max logD(G(z))
    if batch_idx % global_info['gen_k'] == 0:
        disc_fake = disc(fake)
        adversarial_loss = criterion['bce'](disc_fake, torch.ones_like(disc_fake))  # output a positive num
        # content_loss = 0.006 * criterion['vgg_loss'](fake, labels)
        content_loss = 2e-6 * criterion['vgg_loss']((fake + 1.) / 2, (labels + 1.) / 2)
        # content_loss = mse(fake, labels)
        gen_loss = content_loss + global_info['adversarial_weight'] * adversarial_loss + \
                   global_info['pixel_weight'] * criterion['pixel_loss'](fake, labels)
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        log_dict['G_losses'].update(gen_loss.item(), len(inputs))
    # log
    log_dict['F_prob'].update(disc_fake.mean(), 1)
    log_dict['R_prob'].update(disc_real.mean(), 1)
    global_info['t'].set_postfix(loss='Gloss: {:.6f}, Dloss: {:.6f}, fake: {:.2f}, real: {:.2f}'
                                 .format(log_dict['G_losses'].avg, log_dict['D_losses'].avg,
                                         log_dict['F_prob'].avg, log_dict['R_prob'].avg))
    global_info['t'].update(len(inputs))


def train_SRGAN_iter(gen, disc, dataloaders, gen_opt, disc_opt, criterion, global_info, log_dict, use_WGAN=False):
    gen.train()
    disc.train()
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    if use_WGAN:
        iterate = iterate_SRWGAN
    else:
        iterate = iterate_SRGAN
    for batch_idx, data in enumerate(train_dataloader):
        iterate(gen, disc, gen_opt, disc_opt, criterion, batch_idx, data, global_info, log_dict)
        # log per epoch
        if (batch_idx + 1) % global_info['iter_of_epoch'] == 0:
            global_info['step'] += 1
            step = global_info['step']
            # update learning rate
            if global_info['auto_lr'] and step in global_info['milestone']:
                update_lr(gen_opt, 0.1)
                update_lr(disc_opt, 0.1)
                print('learning rate: Gen: {}, Disc: {}\n'.format(get_lr(gen_opt), get_lr(disc_opt)))
            # for multi-GPU
            if isinstance(gen, torch.nn.DataParallel):
                gen2 = gen.module
                disc2 = disc.module
            else:
                gen2 = copy.deepcopy(gen)
                disc2 = copy.deepcopy(disc)
            # validation and log
            global_info['tb_writer']['scalar'].add_scalar('DLoss', log_dict['D_losses'].avg, global_info['step'])
            global_info['tb_writer']['scalar'].add_scalar('GLoss', log_dict['G_losses'].avg, global_info['step'])
            global_info.update({'Gloss': log_dict['G_losses'].avg, 'Dloss': log_dict['D_losses'].avg})
            val_SRGAN(gen2, val_dataloader, global_info)
            # save
            save_GAN_checkpoint_iter(gen2, gen_opt, disc2, disc_opt, global_info)
            # update for next iteration
            log_dict.update({'D_losses': utils.AverageMeter(), 'G_losses': utils.AverageMeter(),
                        'F_prob': utils.AverageMeter(), 'R_prob': utils.AverageMeter()})
            global_info['t'].set_description('step:{}/{}'.format(step, global_info['num_steps'] - 1), refresh=True)


def train_ESRGAN_iter(gen, disc, dataloaders, gen_opt, disc_opt, criterion, global_info, log_dict):
    gen.train()
    disc.train()
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    device = global_info['device']
    for data in train_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        global_info['batch_no'] += 1
        # Train Disciminator  max(logD(x) + log(1-D(G(z))))
        # optimize net_g
        for p in disc.parameters():
            p.requires_grad = False
        fake = gen(inputs)
        # Train Generator min(log(1-D(G(z))) <-> max logD(G(z))
        disc_fake = disc(fake)
        disc_real = disc(labels).detach()
        pixel_loss = global_info['pixel_weight'] * criterion['pixel_loss'](fake, labels)
        gan_loss = global_info['adversarial_weight'] \
             * (criterion['bce'](disc_fake - disc_real.mean(), torch.ones_like(disc_fake))
                + criterion['bce'](disc_real - disc_fake.mean(), torch.zeros_like(disc_real)))/2
        percep_loss = criterion['vgg_loss'](fake, labels)
        gen_loss = percep_loss + gan_loss + pixel_loss
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        for p in disc.parameters():
            p.requires_grad = True
        disc_opt.zero_grad()
        disc_real = disc(labels)
        disc_fake = disc(fake).detach()
        disc_loss_real = 0.5 * criterion['bce'](disc_real - disc_fake.mean(), torch.ones_like(disc_real))
        disc_loss_real.backward()

        disc_fake = disc(fake.detach())
        disc_loss_fake = 0.5 * criterion['bce'](disc_fake - disc_real.detach().mean(), torch.zeros_like(disc_fake))

        disc_loss_fake.backward()
        disc_opt.step()
        with torch.no_grad():
            log_dict['pixel_loss'] = pixel_loss.mean().item()
            log_dict['gan_loss'] = gan_loss.mean().item()
            log_dict['percep_loss'] = percep_loss.mean().item()
            log_dict['G_losses'] = gen_loss.mean().item()
            log_dict['D_losses_real'] = disc_loss_real.mean().item()
            log_dict['D_losses_fake'] = disc_loss_fake.mean().item()
            log_dict['D_losses'] = log_dict['D_losses_real'] + log_dict['D_losses_fake']
            log_dict['F_prob'] = disc_fake.detach().mean().item()
            log_dict['R_prob'] = disc_real.detach().mean().item()
        # log
        global_info['t'].set_postfix(loss='Gloss: {:.6f}, Dloss: {:.6f}, fake: {:.2f}, real: {:.2f}'
                                     .format(log_dict['G_losses'], log_dict['D_losses'],
                                             log_dict['F_prob'], log_dict['R_prob']))
        global_info['t'].update(len(inputs))
        # log per epoch
        if (global_info['batch_no'] + 1) % global_info['iter_of_epoch'] == 0:
            global_info['step'] += 1
            step = global_info['step']
            # update learning rate
            if global_info['auto_lr'] and step in global_info['milestone']:
                update_lr(gen_opt, 0.5)
                update_lr(disc_opt, 0.5)
                print('learning rate: Gen: {}, Disc: {}\n'.format(get_lr(gen_opt), get_lr(disc_opt)))
            # for multi-GPU
            if isinstance(gen, torch.nn.DataParallel):
                gen2 = gen.module
                disc2 = disc.module
            else:
                gen2 = copy.deepcopy(gen)
                disc2 = copy.deepcopy(disc)
            # validation and log
            global_info['tb_writer']['scalar'].add_scalar('DLoss_real', log_dict['D_losses_real'], global_info['step'])
            global_info['tb_writer']['scalar'].add_scalar('DLoss_fake', log_dict['D_losses_fake'], global_info['step'])
            global_info['tb_writer']['scalar'].add_scalar('pixel_loss', log_dict['pixel_loss'], global_info['step'])
            global_info['tb_writer']['scalar'].add_scalar('gan_loss', log_dict['gan_loss'], global_info['step'])
            global_info['tb_writer']['scalar'].add_scalar('percep_loss', log_dict['percep_loss'], global_info['step'])
            global_info['tb_writer']['scalar'].add_scalar('GLoss', log_dict['G_losses'], global_info['step'])
            global_info['tb_writer']['scalar'].add_scalar('DLoss', log_dict['D_losses'], global_info['step'])
            global_info.update({'Gloss': log_dict['G_losses'], 'Dloss': log_dict['D_losses']})
            val_ESRGAN(gen2, val_dataloader, global_info)
            # save
            save_GAN_checkpoint_iter(gen2, gen_opt, disc2, disc_opt, global_info)
            # # update for next iteration
            # log_dict.update({'D_losses': BasicSR_utils.AverageMeter(), 'G_losses': BasicSR_utils.AverageMeter(),
            #                  'D_losses_real': BasicSR_utils.AverageMeter(), 'D_losses_fake': BasicSR_utils.AverageMeter(),
            #                  'pixel_loss': BasicSR_utils.AverageMeter(), 'gan_loss': BasicSR_utils.AverageMeter(), 'percep_loss': BasicSR_utils.AverageMeter(),
            #             'F_prob': BasicSR_utils.AverageMeter(), 'R_prob': BasicSR_utils.AverageMeter()})
            global_info['t'].set_description('step:{}/{}'.format(step, global_info['num_steps'] - 1), refresh=True)
            global_info['batch_no'] = -1


def save_checkpoint(model, optimizer, epoch, epoch_losses, epoch_psnr, best_psnr, best_epoch, outputs_dir, csv_writer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
             'loss': epoch_losses.avg, 'psnr': epoch_psnr.avg, 'best_psnr': best_psnr, 'best_epoch': best_epoch}
    torch.save(state, outputs_dir + f'latest.pth')
    csv_writer.writerow((state['epoch'], state['loss'], state['psnr']))

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        state['best_psnr'] = best_psnr
        state['best_epoch'] = best_epoch
        torch.save(state, outputs_dir + 'best.pth')
    return best_epoch, best_psnr


def save_checkpoint_iter(model, optimizer, global_info, epoch_losses, epoch_psnr):
    step = global_info['step']
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
             'loss': epoch_losses.avg, 'psnr': epoch_psnr.avg, 'best_psnr': global_info['best_psnr'], 'best_step': global_info['best_step']}
    torch.save(state, global_info['outputs_dir'] + f'latest.pth')
    global_info['csv_writer'].writerow((state['step'], state['loss'], state['psnr']))

    if epoch_psnr.avg > global_info['best_psnr']:
        global_info['best_step'] = step
        global_info['best_psnr'] = epoch_psnr.avg
        state['best_psnr'] = epoch_psnr.avg
        state['best_step'] = step
        torch.save(state, global_info['outputs_dir'] + 'best.pth')


def save_GAN_checkpoint(gen, gen_opt, disc, disc_opt, epoch, G_losses, D_losses, epoch_psnr, epoch_niqe, best_niqe, best_epoch, outputs_dir, csv_writer):
    state = {'gen': gen.state_dict(), 'gen_opt': gen_opt.state_dict(),
             'disc': disc.state_dict(), 'disc_opt': disc_opt.state_dict(),
             'epoch': epoch, 'Gloss': G_losses.avg, 'Dloss': D_losses.avg,
             'psnr': epoch_psnr.avg, 'niqe': epoch_niqe.avg, 'best_niqe': best_niqe, 'best_epoch': best_epoch}
    torch.save(state, outputs_dir + f'latest.pth')
    csv_writer.writerow((state['epoch'], state['Gloss'], state['psnr'], state['Dloss'], state['niqe']))

    if epoch_niqe.avg < best_niqe:
        best_epoch = epoch
        best_niqe = epoch_niqe.avg
        state['best_niqe'] = best_niqe
        state['best_epoch'] = best_epoch
        torch.save(state, outputs_dir + f'best_niqe.pth')
    return best_epoch, best_niqe


def save_GAN_checkpoint_iter(gen, gen_opt, disc, disc_opt, global_info):
    state = {'gen': gen.state_dict(), 'gen_opt': gen_opt.state_dict(),
             'disc': disc.state_dict(), 'disc_opt': disc_opt.state_dict(),
             'step': global_info['step'], 'Gloss': global_info['Gloss'], 'Dloss': global_info['Dloss'],
             'psnr': global_info['psnr'], 'niqe': global_info['niqe'],
             'best_niqe': global_info['best_niqe'], 'best_step': global_info['best_step']}
    torch.save(state, global_info['outputs_dir'] + f'latest.pth')
    global_info['csv_writer'].writerow((state['step'], state['Gloss'], state['psnr'], state['Dloss'], state['niqe']))

    if global_info['niqe'] < global_info['best_niqe']:
        global_info['best_step'] = global_info['step']
        global_info['best_niqe'] = global_info['niqe']
        state['best_niqe'] = global_info['best_niqe']
        state['best_step'] = global_info['best_step']
        torch.save(state, global_info['outputs_dir'] + f'best_niqe.pth')
    if (global_info['step']) % 10 == 0:
        state = {'gen': gen.state_dict(), 'psnr': global_info['psnr'], 'niqe': global_info['niqe']}
        torch.save(state, global_info['outputs_dir'] + 'step_{}.pth'.format(global_info['step']))


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