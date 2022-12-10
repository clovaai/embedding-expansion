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

from collections import defaultdict
from torch.utils.data.sampler import Sampler

class RandomSampler(Sampler):
    def __init__(self, dataset, batch_size, num_images_per_id=4, max_num_examples_per_iter=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_images_per_id = num_images_per_id
        self.index_dict = defaultdict(list)
        self.image_id_to_identity_id = {}
        for index, (_, identity_id, category_id, _, _) in enumerate(dataset):
            self.index_dict[identity_id].append(index)
            self.image_id_to_identity_id[index] = identity_id
        self.instance_ids = list(self.index_dict.keys())
        self.num_instance_ids = len(self.instance_ids)
        self.image_ids = list(self.image_id_to_identity_id.keys())
        self.num_image_ids = len(self.image_ids)
        self.max_num_examples_per_iter = max_num_examples_per_iter

    def __iter__(self):
        ret = []
        while len(ret) < self.num_image_ids * self.num_images_per_id:
            indices = np.random.permutation(self.image_ids)
            current_batch_ids = set()
            for i in indices:
                identity_id = self.image_id_to_identity_id[i]
                if identity_id in current_batch_ids:
                    continue
                current_batch_ids.add(identity_id)
                t = self.index_dict[identity_id]
                replace = False if len(t) >= self.num_images_per_id-1 else True
                t = np.random.choice(t, size=self.num_images_per_id-1, replace=replace).tolist()
                ret.extend([i]+t)
                if len(ret) >= self.num_image_ids * self.num_images_per_id or \
                   (self.max_num_examples_per_iter is not None and len(ret) >= self.max_num_examples_per_iter):
                    break
                if len(ret) % self.batch_size == 0:
                    current_batch_ids = set()
            if len(ret) >= self.num_image_ids * self.num_images_per_id or \
               (self.max_num_examples_per_iter is not None and len(ret) >= self.max_num_examples_per_iter):
                break
        return iter(ret)

    def __len__(self):
        val = self.num_image_ids * self.num_images_per_id
        if self.max_num_examples_per_iter is not None:
            val = min(val, self.max_num_examples_per_iter)
        return val
