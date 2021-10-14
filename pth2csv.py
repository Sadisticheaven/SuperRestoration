import csv
import os
import torch

csvFile = open("./SRCNN_x3_lr=1e-02_batch=128.csv", 'w', newline='')

try:
    writer = csv.writer(csvFile)
    writer.writerow(('epoch', 'loss', 'psnr'))

    pthDir = './SRCNN/weight_file/9-1-5_python2/x3/'
    pthList = os.listdir(pthDir)
    if 'best.pth' in pthList:
        pthList.remove('best.pth')
    for pthName in pthList:
        pth = torch.load(pthDir + pthName)
        writer.writerow((pth['epoch'], pth['loss'], pth['psnr']))
finally:
    csvFile.close()
