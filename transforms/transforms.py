'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random

from torchvision import transforms


class Random2DTranslation(object):
    def __init__(self, width, height, p=0.5, interpolation=1, **kwargs):
        self.width = width
        self.height = height
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.random() < self.p:
            return transforms.Resize((self.width, self.height), interpolation=self.interpolation)(img)
        new_width, new_height = int(
            round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = transforms.Resize((new_width, new_height), interpolation=self.interpolation)(img)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = transforms.fixed_crop(resized_img, x1, y1, self.width, self.height)
        return croped_img


class Transforms(object):
    def __init__(self, width=224, height=224, is_train=False):
        if is_train:
            self.T = transforms.Compose([
                         Random2DTranslation(width, height),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])
        else:  # for eval
            self.T = transforms.Compose([
                         transforms.Resize((width, height)),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])

    def __call__(self, x):
        x = self.T(x)
        return x
