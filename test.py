import torch.nn as nn
import torch

#("(3,5,5),(1,1,1),(1,1,1)")
conv3d1 = nn.Conv3d(3, 64, kernel_size=(3,8,8), stride=(1,4,4), padding=(1,2,2))
#("(3,4,4),(1,2,2),(1,2,2)")
conv3d2 = nn.ConvTranspose3d(3, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1),output_padding=(0,1,1))
#反卷积
conv3d21 = nn.ConvTranspose3d(64, 3, kernel_size=(3,8,8), stride=(1,4,4), padding=(1,2,2),output_padding=(0,0,0))
#("(3,3,3),(1,1,1),(1,1,1)")
conv3d3 = nn.Conv3d(3, 64, kernel_size=(3,3,3), stride=1, padding=1)
#1*1 卷积
conv3d4 = nn.Conv3d(6, 3, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0))

tempi0 = torch.rand(8,3,7,128,128)
tempi1 = conv3d1(tempi0)
tempi2 = conv3d2(tempi0)
tempi21 = conv3d21(tempi1)
tempi3 = conv3d3(tempi0)
#tempi4 = torch.cat((tempi0,tempi21),1)
#tempi5 = conv3d4(tempi4)
print(tempi0.shape)
print(tempi1.shape)
print(tempi2.shape)
print(tempi21.shape)
print(tempi3.shape)
#print(tempi4.shape)
#print(tempi5.shape)
print("---------------------------")
