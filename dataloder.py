"""
-*- coding:utf-8 -*-
@Time   :2019/11/27 下午4:08
@Author :wts
@File   :dataloder.py
@Version：1.0
"""
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from skimage import img_as_float
from random import randrange
import os.path
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _DataLoaderIter
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler
import torch
from dataset import MyDataset
#from utils import RandomSampler


class MyDataloader(DataLoader):

    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn = None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None
                 , multiprocessing_context=None, frames_n=3, scale_factor=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.frames_n = frames_n
        self.frames = frames_n * 2 + 1
        self.scale_factor=scale_factor

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise  ValueError('batch_sampler option is mutually exclusive'
                                  'with batch_size, shuffle, sampler, and drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with shuffle')

        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative;'
                             'use num_workers = 0 to disable multiprocessing.')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, key, value):
        if self.__initialized and key in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is'
                             'initialized'.format(key, self.__class__.__name__))

        super(DataLoader, self).__setattr__(key, value)

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)









