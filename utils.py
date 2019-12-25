"""
-*- coding:utf-8 -*-
@Time   :2019/11/27 下午4:54
@Author :wts
@File   :utils.py
@Version：1.0
"""
import torch
import torch.utils.data.sampler as Sampler
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        #print("Mish activation loaded...")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()


