"""
-*- coding:utf-8 -*-
@Time   :2019/11/29 下午3:29
@Author :wts
@File   :model.py
@Version：1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Mish
from net import DenseConv3DDownProjectionUnit, DenseConv3DUpProjectionUnit, PixelShuffle3D, NonLocalBlock3D
from torchsummary import summary
from momonger.memonger import SublinearSequential


class NLBP3D(nn.Module):
    '''
    C3D 测试看看
    '''
    def __init__(self, mid_filters=64):
        super(NLBP3D, self).__init__()

        self.upbili = nn.Upsample(size=[7,256,256], mode='trilinear')

        self.nlb = NonLocalBlock3D(in_channels=3)

        self.convstart = nn.Conv3d(3, mid_filters, kernel_size=(3,3,3), stride=1, padding=1)
        self.d_up1 = DenseConv3DUpProjectionUnit(mid_filters=mid_filters)
        self.d_down1 = DenseConv3DDownProjectionUnit(mid_filters=mid_filters)
        #concat + 1*1卷积
        self.conv1_1_1 = nn.Conv3d(2*mid_filters, mid_filters, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.d_up2 = DenseConv3DUpProjectionUnit(mid_filters=mid_filters)
        # concat + 1*1卷积
        self.conv1_1_2 = nn.Conv3d(64, mid_filters, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.d_down2 = DenseConv3DDownProjectionUnit(mid_filters=mid_filters)
        # concat + 1*1卷积
        self.conv1_1_3 = nn.Conv3d(192, mid_filters, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.d_up3 = DenseConv3DUpProjectionUnit(mid_filters=mid_filters)
        # concat + 1*1卷积
        self.convend = nn.Conv3d(128, 48, kernel_size=3, stride=1, padding=1)
        self.outp = nn.Conv3d(2*mid_filters, 3 ,kernel_size=3, stride=1, padding=1)
        self.upx = nn.Conv3d(3,48,3,1,1)
        self.sps = PixelShuffle3D(scale=4)

        self.relu = Mish()
        self.relu1 = nn.ReLU(inplace=True)

        self.__init__weight()

    def forward(self, x):

        fns = self.nlb(x)
        ###第一块
        #3*3 卷积
        l0 = self.relu(self.convstart(fns))
        h1 = self.d_up1(l0)
        l1 = self.d_down1(h1)
        ######第二块
        #concat l0,l1
        cc1 = torch.cat((l0,l1),1)
        c11_1 = self.relu(self.conv1_1_1(cc1))
        h2 = self.d_up2(c11_1)

        hs = torch.cat((h1,h2),1)
        out = self.relu(self.outp(hs))
        #out = self.relu(self.outp(hs))
        #out = self.sps(hs)
        z = self.upx(x)
        z = self.sps(z)

        return out + z


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
    net = NLBP3D().to(device)
    #net1 = SublinearSequential(NLBP3D(mid_filters=64)).to(device)
    #summary(net1,(3,7,64,64))
    out = net(inputs)
    make_dot(out, params=dict(net.named_parameters()))
    #print(out.shape)




























