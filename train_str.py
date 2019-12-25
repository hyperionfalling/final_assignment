"""
-*- coding:utf-8 -*-
@Time   :2019/12/24 下午4:02
@Author :wts
@File   :train_str.py
@Version：1.0
"""
import torch
import torch.nn as nn
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import NLBP3D
from T3Dtestnet import  Only3DNet

from apex import amp
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MyDataset(dataset_type = '/train', frames_n=3)
dataloder = DataLoader(dataset,batch_size=1,shuffle=False)
net = Only3DNet(mid_filters=64).to(device)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
for epoch in range(100):
    for x, y, zz in dataloder:
        x = torch.tensor(x,requires_grad=True)
        x = x.to(device)
        y = y.to(device)
        zz = zz.to(device)
        #zz = torch.tensor(zz, requires_grad=True)
        out = net(x)
        loss = criterion(out,x)

        print(epoch, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
