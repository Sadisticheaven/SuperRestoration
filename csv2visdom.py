from visdom import Visdom
import csv


def visual_csv(csv_path, viz, line_name):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        loss = []
        epoch = []
        psnr = []
        for i, row in enumerate(reader):
            if i < 1:
                continue
            epoch.append(int(row[0]))
            loss.append(float(row[1]))
            psnr.append(float(row[2]))

    viz.line(Y=loss,
             X=epoch,
             win='Loss',
             name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "loss",  # y轴标签
             })
    viz.line(Y=psnr,
             X=epoch,
             win='PSNR',
             name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "PSNR",  # y轴标签
             })

def visual_csv2(csv_path, viz, line_name):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        loss = []
        epoch = []
        psnr = []
        mean_epoch = 119
        mean_loss = 0.0
        mean_psnr = 0.0
        count = 0
        for i, row in enumerate(reader):
            if i < 1:
                continue
            if i < 119:
                epoch.append(int(row[0]))
                loss.append(float(row[1]))
                psnr.append(float(row[2]))
            else:
                count += 1
                if count == 6:
                    epoch.append(mean_epoch)
                    loss.append(float(row[1]))
                    psnr.append(float(row[2]))
                    mean_epoch += 1
                    count = 0
                else:
                    mean_loss = float(row[1])
                    mean_psnr = float(row[2])


    if not count == 0:
        epoch.append(mean_epoch)
        loss.append(mean_loss)
        psnr.append(mean_psnr)

    viz.line(Y=loss,
             X=epoch,
             win='Loss',
             name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "loss",  # y轴标签
             })
    viz.line(Y=psnr,
             X=epoch,
             win='PSNR',
             name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "PSNR",  # y轴标签
             })

if __name__ == '__main__':
    viz = Visdom(env='FSRCNN')
    visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91_lr=e-2_batch=64/x3/FSRCNN_x3_MSRA_T91_lr=e-2_batch=64.csv', viz=viz, line_name='lr=e-2,batch=64')
    visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91_lr=e-2_batch=128/x3/FSRCNN_x3_MSRA_T91_lr=e-2_batch=128.csv', viz=viz, line_name='lr=e-2,batch=128')
    visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91_lr=e-3_batch=128/x3/FSRCNN_x3_MSRA_T91_lr=e-3_batch=128.csv', viz=viz, line_name='lr=e-3,batch=128')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_56-12-4_MSRA_General191_stride=3_x3_res.csv', viz=viz, line_name='residual')
    # visual_csv('./SRCNN/weight_file/SRCNN_x3_lr=1e-4_batch=128/x3/SRCNN_x3_lr=1e-4_batch=128.csv', viz=viz, line_name='SRCNNx3 lr=1e-4')
    # visual_csv('./SRCNN/weight_file/SRCNN_x3_lr=1e-02_batch=128/x3/SRCNN_x3_lr=1e-02_batch=128.csv', viz=viz, line_name='SRCNNx3 lr=1e-2')
    # visual_csv('./SRCNN/weight_file/9-1-5/SRCNNx3.csv', viz=viz, line_name='SRCNN x3')