import hashlib
import os
import tarfile
import zipfile
import requests

# %matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# @save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):  # 下载一个DATA_HUB中的文件，返回本地文件名
    assert name in DATA_HUB, f"{name}不存在与{DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        """在循环结束后，sha1 对象将包含整个文件的 SHA-1 哈希值。
        这种方式适用于处理大文件，因为它一次只读取一部分数据，避免了一次性将整个文件加载到内存中。"""
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)  # 从文件中读取最多 1048576 字节的数据，并将其赋值给变量 data。这里每次最多读取 1 MB 的数据。
                if not data:  # 如果没有读取到数据，表示文件已经读取完毕，就跳出循环。
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:  # 以十六进制字符串的形式表示
            return fname  # 缓存命中
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  # @save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  # @save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


'''访问和读取数据集'''
DATA_HUB['kaggle_house_train'] = (  # @save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  # @save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

'''数据预处理'''
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]  # ‘shape()’是一个属性，用于获取DataFrame的形状(即行数和列数)，它返回一个包含两个元素的元组，第一个元素表示行数，第二个元素表示列数。
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32  # -1表示行数根据数据维度进行自动的计算
)

"""训练"""
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net=nn.Sequential(nn.Linear(in_features,1))
    return net
