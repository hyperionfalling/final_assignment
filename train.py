"""
-*- coding:utf-8 -*-
@Time   :2019/11/28 下午10:07
@Author :wts
@File   :train.py
@Version：1.0
"""
import torch
import torch.nn as nn
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import NLBP3D
from T3Dtestnet import  Only3DNet
import numpy as np
import math
from apex import amp
from momonger.memonger import SublinearSequential
from torch.autograd import Variable

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MyDataset(dataset_type = '/train', frames_n=3)
dataeval = MyDataset(dataset_type = '/val', frames_n=3)
dataloder = DataLoader(dataset,batch_size=1,shuffle=False)
dataevalloder = DataLoader(dataeval,batch_size=1,shuffle=False)
net = SublinearSequential(NLBP3D(mid_filters=64)).to(device)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
m = 0
for epoch in range(100):
    for x, y, zz in dataloder:
        x = torch.tensor(x,requires_grad=True)
        x = x.to(device)
        y = y.to(device)
        #zz = zz.to(device)
        #zz = torch.tensor(zz, requires_grad=True)
        out = net(x)
        loss = criterion(out,y)

        print(epoch, loss)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    for x, y, zz in dataloder:
        x = x.to(device)
        y = y.to(device)
        zz = zz.to(device)
        #zz = torch.tensor(zz, requires_grad=True)
        out = net(x)


    print(epoch, loss)







