'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .transforms import Transforms

def get_transform(image_size=224):
    height = image_size
    width = image_size
    train_transforms, test_transforms = (
        Transforms(width=width, height=height, is_train=True),
        Transforms(width=width, height=height, is_train=False)
    )
    return (train_transforms, test_transforms)


def get_transform_list():
    return _transforms.keys()
