import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import model_utils
import utils
import os
import torch
from torch.backends import cudnn
from model import G, D
from torch import nn, optim
from SRResNetdatasets import SRResNetValDataset, SRResNetTrainDataset, DIV2KDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train_model(config, from_pth=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['Gpu']

    outputs_dir = config['outputs_dir']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    utils.mkdirs(outputs_dir)
    csv_file = outputs_dir + config['csv_name']
    logs_dir = config['logs_dir']
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])

    # ----需要修改部分------
    print("===> Loading datasets")
    train_dataset = DIV2KDataset(config['train_file'])
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=50)
    # ----END------
    start_epoch, best_epoch, best_psnr, writer, csv_file = \
        model_utils.load_checkpoint(config['weight_file'], model, optimizer, csv_file,
                                    from_pth, auto_lr=config['auto_lr'])

    if torch.cuda.device_count() > 1:
        print("Using GPUs.\n")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    writer_scalar = SummaryWriter(f"{logs_dir}/scalar")
    writer_test = SummaryWriter(f"{logs_dir}/test")

    for epoch in range(start_epoch, num_epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'learning rate: {lr}\n')
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description(f'epoch:{epoch}/{num_epochs - 1}')
            epoch_losses = model_utils.train(model, train_dataloader, optimizer, criterion, device, t)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.eval()
        epoch_psnr = utils.AverageMeter()
        for idx, data in enumerate(val_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            with torch.no_grad():
                preds = model(inputs) * 0.5 + 0.5
                img_grid_fake = torchvision.utils.make_grid(preds, normalize=True)
                writer_test.add_image(f"Test Fake{idx}", img_grid_fake, global_step=epoch)
            preds = preds.mul(255.0).cpu().numpy().squeeze(0)
            preds = preds.transpose([1, 2, 0])  #chw->hwc
            preds = np.clip(preds, 0.0, 255.0)
            preds = utils.rgb2ycbcr(preds).astype(np.float32)[..., 0]/255.
            epoch_psnr.update(utils.calc_psnr(preds, labels.numpy()[0, 0, ...]).item(), len(inputs))
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        if config['auto_lr']:
            scheduler.step(epoch_psnr.avg)

        writer_scalar.add_scalar('PSNR', epoch_psnr.avg, epoch)
        writer_scalar.add_scalar('Loss', epoch_losses.avg, epoch)

        best_epoch, best_psnr = model_utils.save_checkpoint(model, optimizer, epoch, epoch_losses,
                                                            epoch_psnr, best_psnr, best_epoch, outputs_dir, writer)
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))