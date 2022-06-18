import os
from PIL import Image
import lmdb
import numpy as np
from tqdm import tqdm

root_dirs = ['./datasets/Set5/']
lmdb_save_path = './datasets/Set5.lmdb'
# 创建数据库文件
img_list = []
for root_dir in root_dirs:
    img_names = os.listdir(root_dir)
    for name in img_names:
        img_list.append(root_dir + name)
dataset = []
data_size = 0

print('Read images...')
with tqdm(total=len(img_list)) as t:
    for i, v in enumerate(img_list):
        img = np.array(Image.open(v).convert('RGB'))
        dataset.append(img)
        data_size += img.nbytes
        t.update(1)
env = lmdb.open(lmdb_save_path, max_dbs=2, map_size=data_size * 2)
print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))
# 创建对应的数据库
train_data = env.open_db("train_data".encode('ascii'))
train_shape = env.open_db("train_shape".encode('ascii'))
# 把图像数据写入到LMDB中
with env.begin(write=True) as txn:
    with tqdm(total=len(dataset)) as t:
        for idx, img in enumerate(dataset):
            H, W, C = img.shape
            txn.put(str(idx).encode('ascii'), img, db=train_data)
            meta_key = (str(idx) + '.meta').encode('ascii')
            meta = '{:d}, {:d}, {:d}'.format(H, W, C)
            txn.put(meta_key, meta.encode('ascii'), db=train_shape)
            t.update(1)
env.close()
print('Finish writing lmdb.')