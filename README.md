#Final Assignment

##1.Introduction

基于RBPN（CVPR2019）的修改的用以视频超分辨网络模型

##2.Content

###2.1模型介绍

**RBPN**是**DBPN**（2018）的视频改进版，采用光流对齐，多帧图像的渐进融合，
内部采用了BDPNs块进行递归的图像超分，但是损失函数和其对上下帧之间的信息
利用还不高，可以加以改进。

###2.2改进

~~1.引入non-local network（类似于注意力机制）~~

2.加入GAN损失、感知损失
    
