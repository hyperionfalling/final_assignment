"""
-*- coding:utf-8 -*-
@Time   :2019/11/26 下午1:25
@Author :wts
@File   :dataset.py
@Version：1.0
"""
import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from skimage import img_as_float
from random import randrange
import matplotlib.pyplot as plt
import os.path
import cv2
from torch.utils.data import Dataset
import random

class MyDataset(Dataset):
    '''an abstract class representing'''
    def __init__(self, dataset_type, transform=None, update_dataset=False, frames_n = 3):
        '''
        :param dataset_type: ['train','test']
        :param transform:
        :param update_dataset:
        '''
        dataset_path = '/home/wts/practice/dlpyprac/testmodel/dataset'
        self.frames_n = frames_n
        self.frames = frames_n * 2 + 1

        if update_dataset:
            print("update dataset")
            dbtype_list = os.listdir(dataset_path)
            #dbtype_list.remove('datalist.txt')
            for dbtype in dbtype_list:
                each_path = os.path.join(dataset_path,dbtype)
                each_list = os.listdir(each_path)
                f = open(each_path + "/datalist.txt","w")
                each_list.remove('datalist.txt')
                for each_db in each_list:
                    each_sum_name = os.path.join(each_path, each_db)
                    IG_name = os.listdir(each_sum_name)
                    img_m_path0 = os.path.join(each_sum_name,IG_name[0])
                    img_m_path1 = os.path.join(each_sum_name, IG_name[1])
                    img_name = os.listdir(img_m_path0)
                    img_name = sorted(img_name)
                    for img in img_name:
                        temp = os.path.join(img_m_path0,img)
                        f.write(temp)
                        f.write('*')
                        temp = os.path.join(img_m_path1, img)
                        f.write(temp)
                        f.write('\n')
                f.close()

        self.transform = transform
        self.sample_list = list()
        self.dataset_type = dataset_type
        f = open(dataset_path + self.dataset_type + '/datalist.txt')
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()


    def __getitem__(self, index):
        c_ind = index % 32
        if(c_ind < self.frames_n):
            index += self.frames_n
        elif(c_ind > 31 - self.frames_n):
            index -= self.frames_n

        sht = self.sample_list[index]
        imgs = Image.open(sht.split('*')[-1]).convert('RGB')
        imgs = np.array(imgs)
        img_frames = torch.empty((self.frames,3,64,64))
        label_frames = torch.empty((self.frames,3,256,256))
        img_bi_frames = torch.empty((self.frames, 3, 256, 256))
        hi = random.randint(0,(imgs.shape[1]-64))
        wi = random.randint(0,(imgs.shape[0]-64))
        hl = 4 * hi
        wl = 4 * wi
        t = 0
        for i in range(index-self.frames_n,index+self.frames_n+1):
            item = self.sample_list[i]
            img = Image.open(item.split('*')[-1]).convert('RGB')
            label = Image.open(item.split('*')[0]).convert('RGB')
            img = img.crop((hi,wi,hi+64,wi+64))
            size = img.size
            new_size = tuple([int (x * 4) for x in size])
            img_bi = img.resize(new_size, resample=Image.BICUBIC)
            label = label.crop((hl,wl,hl+256,wl+256))
            img = np.array(img)
            label = np.array(label)
            img_bi = np.array(img_bi)
            img = np.atleast_3d(img).transpose(2,0,1).astype(np.float32)
            label = np.atleast_3d(label).transpose(2,0,1).astype(np.float32)
            img_bi = np.atleast_3d(img_bi).transpose(2,0,1).astype(np.float32)
            img = torch.from_numpy(img).float()
            label = torch.from_numpy(label).float()
            img_bi = torch.from_numpy(img_bi).float()
            img_frames[t] = img
            label_frames[t]= label
            img_bi_frames[t] = img_bi
            t += 1
        if self.transform is not None:
            img = self.transform(img)
            label = self.transpose(label)

        img_frames = img_frames.transpose(0,1)
        label_frames = label_frames.transpose(0,1)
        img_bi_frames = img_bi_frames.transpose(0,1)

        return img_frames, label_frames, img_bi_frames
        #return img, label


    def __len__(self):
        return len(self.sample_list)

    def make_txt_file(self, path):
        return path

if __name__ == '__main__':
    import torch.nn as nn
    ds = MyDataset(dataset_type = '/train')
    img, gt, bic = ds.__getitem__(3)
    print(img.shape)
    img1 = img.permute(1,2,3,0)
    gt1 = gt.permute(1,2,3,0)
    bic1 = bic.permute(1,2,3,0)
    img2 = img1.numpy()
    gt2 = gt1.numpy()
    bic2 = bic1.numpy()
    plt.subplot(271)
    plt.imshow(img2[0].astype(int))
    plt.subplot(272)
    plt.imshow(gt2[0].astype(int))
    plt.subplot(273)
    plt.imshow(img2[1].astype(int))
    plt.subplot(274)
    plt.imshow(gt2[1].astype(int))
    plt.subplot(275)
    plt.imshow(img2[2].astype(int))
    plt.subplot(276)
    plt.imshow(gt2[2].astype(int))
    plt.subplot(277)
    plt.imshow(img2[3].astype(int))
    plt.subplot(2,7,8)
    plt.imshow(gt2[3].astype(int))
    plt.subplot(2,7,9)
    plt.imshow(img2[4].astype(int))
    plt.subplot(2,7,10)
    plt.imshow(gt2[4].astype(int))
    plt.subplot(2,7,11)
    plt.imshow(img2[5].astype(int))
    plt.subplot(2,7,12)
    plt.imshow(gt2[5].astype(int))
    plt.subplot(2, 7, 13)
    plt.imshow(img2[6].astype(int))
    plt.subplot(2, 7, 14)
    plt.imshow(gt2[6].astype(int))

    plt.show()


























