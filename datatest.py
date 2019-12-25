"""
-*- coding:utf-8 -*-
@Time   :2019/11/27 下午7:30
@Author :wts
@File   :datatest.py
@Version：1.0
"""
import numpy as np
import torch
from dataset import MyDataset
#from dataloader import MyDataloader
from torch.utils.data import DataLoader

dataset = MyDataset(dataset_type = '/train', frames_n=2)
dataloder = DataLoader(dataset,batch_size=8,shuffle=False)
i = 0
for x, y in dataloder:
    if(i % 10 == 0):
        print(x.shape)
        print(y.shape)
        print(i)
    i += 1
