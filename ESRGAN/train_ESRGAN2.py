import torchvision
from tqdm import tqdm
import model_utils
import utils
import os
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from VGGLoss2 import VGGLoss
from model import G, D
from torch import nn, optim
from ESRGANdatasets import ESRGANValDataset, ESRGANTrainDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import niqe
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train_model(config, pre_train, from_pth=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['Gpu']
    outputs_dir = config['outputs_dir']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    utils.mkdirs(outputs_dir)
    csv_file = outputs_dir + config['csv_name']
    logs_dir = config['logs_dir']
    gen_k = config['gen_k']
    disc_k = config['disc_k']
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])

    # ----需要修改部分------
    print("===> Loading datasets")
    train_dataset = ESRGANTrainDataset()
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=config['num_workers'],
                                  batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataset = ESRGANValDataset(config['val_file'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    print("===> Building model")
    gen = G()
    disc = D()
    # bce = nn.BCEWithLogitsLoss().to(device)
    bce = nn.BCELoss().to(device)
    l1 = nn.L1Loss().to(device)
    vgg_loss = VGGLoss(device)
    gen_opt = optim.Adam(gen.parameters(), lr=config['gen_lr'])
    disc_opt = optim.Adam(disc.parameters(), lr=config['disc_lr'])
    # ----END------
    start_epoch, best_epoch, best_niqe, writer, csv_file = \
        model_utils.load_GAN_checkpoint(pre_train, config['weight_file'], gen, gen_opt, disc, disc_opt,
                                        csv_file, from_pth, config['auto_lr'])

    if torch.cuda.device_count() > 1:
        print("Using GPUs.\n")
        gen = torch.nn.DataParallel(gen)
        disc = torch.nn.DataParallel(disc)
    gen = gen.to(device)
    disc = disc.to(device)

    writer_scalar = SummaryWriter(f"{logs_dir}/scalar")
    writer_test = SummaryWriter(f"{logs_dir}/test")

    adversarial_weight = config['adversarial_weight']
    pixel_weight = config['pixel_weight']
    for epoch in range(start_epoch, num_epochs):
        print('learning rate: Gen: {}, Disc: {}\n'.format(model_utils.get_lr(gen_opt), model_utils.get_lr(disc_opt)))
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description(f'epoch:{epoch}/{num_epochs - 1}')
            gen.train()
            disc.train()
            D_losses = utils.AverageMeter()
            G_losses = utils.AverageMeter()
            F_prob = utils.AverageMeter()
            R_prob = utils.AverageMeter()
            for batch_idx, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Train Disciminator  max(logD(x) + log(1-D(G(z))))
                fake = gen(inputs)
                if batch_idx % disc_k == 0:
                    disc_real = disc(labels)
                    disc_fake = disc(fake.detach())
                    disc_loss_real = bce(torch.sigmoid(disc_real - disc_fake.mean()), torch.ones_like(disc_real))
                    disc_loss_fake = bce(torch.sigmoid(disc_fake - disc_real.mean()), torch.zeros_like(disc_fake))
                    disc_loss = (disc_loss_real + disc_loss_fake) / 2
                    disc_opt.zero_grad()
                    disc_loss.backward()
                    disc_opt.step()
                    D_losses.update(disc_loss.item(), len(inputs))
                else:
                    with torch.no_grad():
                        disc_real = torch.sigmoid(disc(labels))
                        disc_fake = torch.sigmoid(disc(fake.detach()))
                # Train Generator min(log(1-D(G(z))) <-> max logD(G(z))
                if batch_idx % gen_k == 0:
                    disc_fake = disc(fake)
                    disc_real = disc(labels)
                    pixel_loss = l1(fake, labels)
                    adversarial_loss =(bce(torch.sigmoid(disc_fake - disc_real.mean()), torch.ones_like(disc_fake)) + \
                                       bce(torch.sigmoid(disc_real - disc_fake.mean()), torch.zeros_like(disc_real)))/2
                    percep_loss = vgg_loss(fake, labels)
                    gen_loss = percep_loss + adversarial_weight * adversarial_loss + pixel_weight * pixel_loss
                    gen_opt.zero_grad()
                    gen_loss.backward()
                    gen_opt.step()
                    G_losses.update(gen_loss.item(), len(inputs))
                # log
                F_prob.update((disc_fake - disc_real.mean()).mean(), 1)
                R_prob.update((disc_real - disc_fake.mean()).mean(), 1)
                t.set_postfix(loss='Gloss: {:.6f}, Dloss: {:.6f}, fake: {:.2f}, real: {:.2f}'
                              .format(G_losses.avg, D_losses.avg, F_prob.avg, R_prob.avg))
                t.update(len(inputs))
        writer_scalar.add_scalar('DLoss', D_losses.avg, epoch)
        writer_scalar.add_scalar('GLoss', G_losses.avg, epoch)
        if isinstance(gen, torch.nn.DataParallel):
            gen = gen.module
            disc = disc.module
        gen.eval()
        epoch_psnr = utils.AverageMeter()
        epoch_niqe = utils.AverageMeter()
        for idx, data in enumerate(val_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            with torch.no_grad():
                fake = gen(inputs)
            img_grid_fake = torchvision.utils.make_grid(fake.mul(255.0), normalize=True)
            writer_test.add_image(f"Test Fake{idx}", img_grid_fake, global_step=epoch, dataformats='CHW')
            fake = fake.mul(255.0).cpu().numpy().squeeze(0)
            fake = fake.transpose([1, 2, 0])  # chw->hwc
            fake = np.clip(fake, 0.0, 255.0)
            fake = utils.rgb2ycbcr(fake).astype(np.float32)[..., 0] / 255.
            epoch_niqe.update(niqe.calculate_niqe(fake), len(inputs))
            epoch_psnr.update(utils.calc_psnr(fake, labels.numpy()[0, 0, ...]).item(), len(inputs))
        print('eval psnr: {:.2f}, niqe: {:.4f}'.format(epoch_psnr.avg, epoch_niqe.avg))
        writer_scalar.add_scalar('PSNR', epoch_psnr.avg, epoch)
        writer_scalar.add_scalar('NIQE', epoch_niqe.avg, epoch)
        if config['auto_lr'] and epoch in [58, 116, 232, 348]:
            model_utils.update_lr(gen_opt, 0.5)
            model_utils.update_lr(disc_opt, 0.5)
        best_epoch, best_niqe = model_utils.save_GAN_checkpoint(
            gen, gen_opt, disc, disc_opt, epoch, G_losses, D_losses,
            epoch_psnr, epoch_niqe, best_niqe, best_epoch, outputs_dir, writer)
    print('best epoch: {}, niqe: {:.4f}'.format(best_epoch, best_niqe))
