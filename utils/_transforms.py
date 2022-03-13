from __future__ import absolute_import

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import *

import numpy as np
import torch
import random
import math


# Prepare transform processor
"""
# transforms.Compose()类用来组合多个torchvision.transforms操作
# 一个list数组，数组里是多个'Transform'对象
# 遍历list数组，对img依次执行每个transforms操作，并返回transforms后的img
"""  
transform_dict = {'train': transforms.Compose([transforms.RandomResizedCrop(224), # RandomResizedCrop 将给定图像随机裁剪为不同的大小和宽高比,然后缩放所裁剪得到的图像为制定的大小
                                               transforms.RandomHorizontalFlip(), # RandomHorizontalFlip 以给定的概率随机水平翻转给定的PIL图像
                                               transforms.ToTensor(), # ToTensor 把PIL.Image或者numpy.narray数据类型转变为torch.FloatTensor类型
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), # Normalize  对每个通道而言执行操作image=(image-mean)/std
                  'test': transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])}


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class Rotation(object):
    def __init__(self, degree):
        self.degree = degree
       
    def __call__(self, img):
        img = img.rotate(self.degree)
        return img

class Crop_img(object):
    def __init__(self, the_type, size):
        self.type = the_type
        self.size = size
       
    def __call__(self, img):
        origin_size = img.size[1]
        offset = origin_size - self.size
        half_offset = offset/2
        crop_img = img
        if self.type == 'Center':
            area = (half_offset, half_offset, origin_size-half_offset, origin_size-half_offset)
            crop_img = img.crop(area)
        elif self.type == 'LeftTop':
            area = (0, 0, self.size, self.size)
            crop_img = img.crop(area)
        elif self.type == 'RightDown':
            area = (offset, offset, origin_size, origin_size)
            crop_img = img.crop(area)
        elif self.type == 'LeftDown':
            area = (0, offset, self.size, origin_size)
            crop_img = img.crop(area)
        elif self.type == 'RightTop':
            area = (offset, 0, origin_size, self.size)
            crop_img = img.crop(area)
        return crop_img