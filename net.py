"""
-*- coding:utf-8 -*-
@Time   :2019/11/30 下午1:50
@Author :wts
@File   :net.py
@Version：1.0
"""
import torch
from torch import nn
from torch.nn import functional as F
from utils import Mish
import numpy as np
import torchsummary as summary

class NonLocalBlock3D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True):
        super(NonLocalBlock3D,self).__init__()

        assert dimension in [3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_3d = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.g = conv_3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = conv_3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = conv_3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x:(b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class DenseConv3DUpProjectionUnit(nn.Module):
    def __init__(self, mid_filters=32):
        super(DenseConv3DUpProjectionUnit, self).__init__()
        #1*1 卷积将RGB三通道转为64通道，7帧保持不变
        self.l2h3ddeconv1 = nn.ConvTranspose3d(mid_filters, mid_filters, kernel_size=(3,8,8), stride=(1,4,4),padding=(1,2,2),
                                               output_padding=0)
        self.l2h3dconv1 = nn.Conv3d(mid_filters, mid_filters, kernel_size=(3,8,8), stride=(1,4,4), padding=(1,2,2))
        self.el2h3ddeconv2 = nn.ConvTranspose3d(mid_filters, mid_filters, kernel_size=(3,8,8), stride=(1,4,4), padding=(1,2,2),
                                               output_padding=0)
        self.relu = Mish()

    def forward(self, x):
        h0 = self.relu(self.l2h3ddeconv1(x))
        l0 = self.relu(self.l2h3dconv1(h0))
        el = x - l0
        eh = self.relu(self.el2h3ddeconv2(el))
        h1 = h0 + eh

        return h1


class DenseConv3DDownProjectionUnit(nn.Module):
    def __init__(self,mid_filters=32):
        super(DenseConv3DDownProjectionUnit, self).__init__()
        # 1*1 卷积将RGB三通道转为64通道，7帧保持不变
        self.h2l3dconv1 = nn.Conv3d(mid_filters, mid_filters, kernel_size=(3, 8, 8), stride=(1, 4, 4), padding=(1, 2, 2))
        self.h2l3ddeconv1 = nn.ConvTranspose3d(mid_filters, mid_filters, kernel_size=(3, 8, 8), stride=(1, 4, 4), padding=(1, 2, 2),
                                             output_padding=0)
        self.eh2l3dconv2 = nn.Conv3d(mid_filters, mid_filters, kernel_size=(3, 8, 8), stride=(1, 4, 4), padding=(1, 2, 2))
        self.relu = Mish()

    def forward(self, x):
        l0 = self.relu(self.h2l3dconv1(x))
        h0 = self.relu(self.h2l3ddeconv1(l0))
        eh = x - h0
        el = self.relu(self.eh2l3dconv2(eh))
        l1 = l0 + el

        return l1

class PixelShuffle3D(nn.Module):
    def __init__(self, scale):
        '''

        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 2

        out_depth = in_depth
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        inpuy_view = input.contiguous().view(batch_size, nOut, 1, self.scale, self.scale, in_depth,
                                             in_height, in_width)

        output = inpuy_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


if __name__ == '__main__':
    import torch

    img = torch.randn(2, 32, 5, 4, 4)
    net = DenseConv3DUpProjectionUnit()
    summary(net,(3,5,4,4))
    #out = net(img)
    #print(out.size())














