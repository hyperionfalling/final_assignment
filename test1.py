"""
-*- coding:utf-8 -*-
@Time   :2019/12/7 下午3:40
@Author :wts
@File   :test1.py
@Version：1.0
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("000.png").convert('RGB')
img = np.array(img)
img = np.atleast_3d(img).transpose(2,0,1).astype(np.float32)
img = torch.from_numpy(img).float()
img = torch.unsqueeze(img,1)
img = torch.unsqueeze(img,0)


print(img.shape)
up1 = nn.Upsample(size=[1, 360, 636], mode='trilinear')
out = up1(img)
out = torch.squeeze(out)
print(out.shape)
out = out.numpy()
img1 = out.transpose(1,2,0)
print(img1.shape)
plt.imshow(img1.astype(int))
plt.show()

