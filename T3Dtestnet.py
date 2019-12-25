"""
-*- coding:utf-8 -*-
@Time   :2019/12/24 上午10:14
@Author :wts
@File   :3Dtestnet.py
@Version：1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Mish
from net import DenseConv3DDownProjectionUnit, DenseConv3DUpProjectionUnit, PixelShuffle3D, NonLocalBlock3D
from torchsummary import summary
from momonger.memonger import SublinearSequential

class Only3DNet(nn.Module):
    def __init__(self, mid_filters=64):
        super(Only3DNet, self).__init__()
        #上采样层 无参数
        self.upbili = nn.Upsample(size=[7, 256, 256], mode='trilinear')

        #feature层
        self.feat3d1 = nn.Conv3d(3, mid_filters, kernel_size=3, stride=1, padding=1)

        #重建层
        self.cons1 = nn.Conv3d(mid_filters, 48, kernel_size=3, stride=1, padding=1)
        self.sps = PixelShuffle3D(scale=4)

        self.relu = Mish()
        self.__init__weight()

    def forward(self, x):
        out1 = self.upbili(x)
        feat0 = self.relu(self.feat3d1(x))
        cons1 = self.relu(self.cons1(feat0))
        out2 = self.relu(self.sps(cons1))

        out = out1 + out2

        return out

    def __init__weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

if __name__ == "__main__":

    from torchviz import make_dot, make_dot_from_trace
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.rand(1,3,7,64,64).to(device)
    inputs = torch.tensor(inputs,requires_grad=True)
    #net = Only3DNet().to(device)
    net1 = SublinearSequential(Only3DNet(mid_filters=64)).to(device)
    summary(net1,(3,7,64,64))
    #out = net(inputs)
    #make_dot(out, params=dict(net.named_parameters()))
    #print(out.shape)

