from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import model_utils
import utils
import os
import torch
from torch.backends import cudnn
from torch import nn, optim
from model import N2_10_4, CLoss, HuberLoss
from FSRCNNdatasets import TrainDataset, ValDataset, ResValDataset
from torch.utils.data.dataloader import DataLoader


def train_model(config, from_pth=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['Gpu']
    outputs_dir = config['outputs_dir']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    csv_file = outputs_dir + config['csv_name']
    lr = config['lr']
    logs_dir = config['logs_dir']
    utils.mkdirs(outputs_dir)
    utils.mkdirs(logs_dir)
    weight_decay = config['weight_decay']

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])
    # ----需要修改部分------
    model = N2_10_4(config['scale'], config['in_size'], config['out_size'],
                    num_channels=1, d=config['d'], m=config['m'])
    if not from_pth:
        model.init_weights(method=config['init'])

    if config['Loss'] == 'CLoss':
        criterion = CLoss(delta=config['delta']).cuda()
    elif config['Loss'] == 'Huber':
        criterion = HuberLoss(delta=config['delta']).cuda()
    else:
        criterion = nn.MSELoss().cuda()

    # extract_weight = [n.weight for n in model.extract_layer if isinstance(n, nn.Conv2d)]
    # extract_PReLU = [n.weight for n in model.extract_layer if isinstance(n, nn.PReLU)]
    # extract_bias = [n.bias for n in model.extract_layer if isinstance(n, nn.Conv2d)]
    # mid_weight = [n.weight for n in model.mid_part if isinstance(n, nn.Conv2d)]
    # mid_PReLU = [n.weight for n in model.mid_part if isinstance(n, nn.PReLU)]
    # mid_bias = [n.bias for n in model.mid_part if isinstance(n, nn.Conv2d)]
    # deconv_weight = model.deconv_layer.weight
    # deconv_bias = model.deconv_layer.bias
    # optimizer = optim.SGD([
    #     {'params': extract_weight, 'weight_decay': weight_decay},
    #     {'params': extract_PReLU},
    #     {'params': extract_bias, 'lr': lr},
    #
    #     {'params': mid_weight, 'weight_decay': weight_decay},
    #     {'params': mid_PReLU},
    #     {'params': mid_bias, 'lr': lr},
    #
    #     {'params': deconv_weight, 'weight_decay': weight_decay, 'lr':  lr * 0.1},
    #     {'params': deconv_bias, 'lr': lr * 0.1},
    optimizer = optim.SGD([
        {'params': model.extract_layer.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.deconv_layer.parameters(), 'lr': lr * 0.1},
    ], lr=lr, momentum=0.9)  # 前两层学习率lr， 最后一层学习率lr*0.1

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
                                    from_pth, config['auto_lr'])

    if torch.cuda.device_count() > 1:
        print("Using GPUs.\n")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    writer_scalar = SummaryWriter(f"{logs_dir}/scalar")

    for epoch in range(start_epoch, num_epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'learning rate: {lr}\n')
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description(f'epoch:{epoch}/{num_epochs - 1}')
            epoch_losses = model_utils.train(model, train_dataloader, optimizer, criterion, device, t)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        epoch_psnr = model_utils.validate(model, val_dataloader, device, config['residual'])

        writer_scalar.add_scalar('Loss', epoch_losses.avg, epoch)
        writer_scalar.add_scalar('PSNR', epoch_psnr.avg, epoch)

        best_epoch, best_psnr = model_utils.save_checkpoint(model, optimizer, epoch, epoch_losses,
                                                            epoch_psnr, best_psnr, best_epoch, outputs_dir, writer)
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))