'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .sampler import RandomSampler
from .dataset import Dataset, ImageData

import torch
import os


def get_train_data_loader(dataset, num_identities, train_transform, batch_size, num_instances, num_workers):
    sampler = RandomSampler(dataset,
                            batch_size,
                            num_instances,
                            max_num_examples_per_iter=num_instances*num_identities)

    train_loader = torch.utils.data.DataLoader(
        ImageData(dataset, train_transform),
        sampler=sampler,
        batch_size=batch_size, num_workers=num_workers, drop_last=True
    )
    return train_loader


def get_test_data_loader(dataset, test_transform, batch_size, num_workers):
    test_loader = torch.utils.data.DataLoader(
        ImageData(dataset, test_transform),
        batch_size=batch_size, num_workers=num_workers,
    )
    return test_loader


def get_data_loader(data_dir, train_meta, test_meta, train_transform, test_transform, batch_size, num_instances, num_workers):
    dataset = Dataset(data_dir, train_meta, test_meta)
    dataset.print_stats()

    train_loader = get_train_data_loader(dataset.train,
                                        dataset.num_train_ids,
                                        train_transform,
                                        batch_size,
                                        num_instances,
                                        num_workers)
    
    test_loader = get_test_data_loader(dataset.test, test_transform, batch_size, num_workers)

    return train_loader, test_loader
