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
    viz.line(Y=loss, X=epoch, win='Loss', name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "loss",  # y轴标签
             })
    viz.line(Y=psnr, X=epoch, win='PSNR', name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "PSNR",  # y轴标签
             })


def visual_csv2(csv_path, viz, line_name):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        Gloss = []
        Dloss = []
        epoch = []
        psnr = []
        for i, row in enumerate(reader):
            if i < 1:
                continue
            epoch.append(int(row[0]))
            Gloss.append(float(row[1]))
            psnr.append(float(row[2]))
            Dloss.append(float(row[3]))

    viz.line(Y=Gloss, X=epoch, win='GLoss', name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "Gloss",  # y轴标签
             })
    viz.line(Y=Dloss, X=epoch, win='DLoss', name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "Dloss",  # y轴标签
             })
    viz.line(Y=psnr, X=epoch, win='PSNR', name=line_name,  # 线条名称
             update='append',  # 以添加方式加入
             opts={
                 'showlegend': True,  # 显示网格
                 'xlabel': "epoch",  # x轴标签
                 'ylabel': "PSNR",  # y轴标签
             })

if __name__ == '__main__':
    viz = Visdom(env='FSRCNN')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91_lr=e-2_batch=64_input=9/x3/FSRCNN_x3_MSRA_T91_lr=e-2_batch=64_input=9.csv', viz=viz, line_name='baseline')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91_lr=e-1_batch=128_input=9/x3/FSRCNN_x3_MSRA_T91_lr=e-1_batch=128_input=9.csv', viz=viz, line_name='9-19')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_XavierTanh_T91_lr=e-1_batch=64_input=11/x3/FSRCNN_x3_XavierTanh_T91_lr=e-1_batch=64_input=11.csv', viz=viz, line_name='Xavier Tanh')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91_lr=e-1_batch=128_input=11/x3/FSRCNN_x3_MSRA_T91_lr=e-1_batch=128_input=11.csv', viz=viz, line_name='baseline')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91_lr=e-1_batch=128_out=27/x3/FSRCNN_x3_MSRA_T91_lr=e-1_batch=128_out=27.csv', viz=viz, line_name='11-27')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91res_lr=e-1_batch=128_input=11/x3/FSRCNN_x3_MSRA_T91res_lr=e-1_batch=128_input=11.csv', viz=viz, line_name='91residual-Modify')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_T91resMod_lr=e-1_batch=128_input=11/x3/FSRCNN_x3_MSRA_T91resMod_lr=e-1_batch=128_input=11.csv', viz=viz, line_name='91residual')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_MSRA_G191resMod_lr=1e-1_batch=128_input=11/x3/FSRCNN_x3_MSRA_G191resMod_lr=1e-1_batch=128_input=11.csv', viz=viz, line_name='191residual')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_x3_Xavier_T91_lr=e-2_batch=64_input=11/x3/FSRCNN_x3_Xavier_T91_lr=e-2_batch=64_input=11.csv', viz=viz, line_name='Xavier PReLU')
    # visual_csv('./FSRCNN/weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_out=19/x3/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_out=19.csv', viz=viz, line_name='N2-10-4')
    # visual_csv('./FSRCNN/weight_file/N2-10-4_x3_MSRA_G191res_lr=5e-1_batch=512_out=19/x3/N2-10-4_x3_MSRA_G191res_lr=5e-1_batch=512_out=19.csv', viz=viz, line_name='N2-10-4 G191')
    # visual_csv('./FSRCNN/weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_CLoss=e-2/x3/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_CLoss=e-2.csv', viz=viz, line_name='CLoss delta=1e-2')
    # visual_csv('./FSRCNN/weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_CLoss=e-3/x3/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_CLoss=e-3.csv', viz=viz, line_name='CLoss delta=1e-3')
    # visual_csv('./FSRCNN/weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_CLoss=e-4/x3/N2-10-4_x3_MSRA_T91res_lr=e-1_batch=128_CLoss=e-4.csv', viz=viz, line_name='CLoss delta=1e-4')
    # visual_csv('./FSRCNN/weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=1_batch=128_Huber=6e-1/x3/N2-10-4_x3_MSRA_T91res_lr=1_batch=128_Huber=6e-1.csv', viz=viz, line_name='Huber delta=0.6')
    # visual_csv('./FSRCNN/weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=1_batch=128_Huber=8e-1/x3/N2-10-4_x3_MSRA_T91res_lr=1_batch=128_Huber=8e-1.csv', viz=viz, line_name='Huber delta=0.8')
    # visual_csv('./FSRCNN/weight_file/Test4/N2-10-4_x3_MSRA_T91res_lr=1_batch=128_Huber=9e-1/x3/N2-10-4_x3_MSRA_T91res_lr=1_batch=128_Huber=9e-1.csv', viz=viz, line_name='Huber delta=0.9')
    # visual_csv('./FSRCNN/weight_file/FSRCNN_56-12-4_MSRA_General191_stride=3_x3_res.csv', viz=viz, line_name='residual')
    # visual_csv('./SRResNet/weight_file/SRResNet_x4_MSRA_291_lr=e-4_batch=16_out=96/x4/SRResNet_x4_MSRA_291_lr=e-4_batch=16_out=96.csv', viz=viz, line_name='lr=e-4 batch=16')
    # visual_csv('./SRResNet/weight_file/SRResNet_x4_MSRA_DIV2Kaug_lr=e-4_batch=16_out=96/x4/SRResNet_x4_MSRA_DIV2Kaug_lr=e-4_batch=16_out=96.csv', viz=viz, line_name='DIV2K')
    visual_csv2('./SRResNet/weight_file/SRGAN_x4_MSRA_DIV2Kaug_lr=e-4_batch=16_out=96/x4/SRGAN_x4_MSRA_DIV2Kaug_lr=e-4_batch=16_out=96.csv', viz=viz, line_name='DIV2K')
