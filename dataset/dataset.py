'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import torchvision
import PIL.Image

from torch.utils.data import Dataset


class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        image, instance_id, category_id, view_id, bbox = self.dataset[item]        
        image = PIL.Image.open(image)

        if bbox is not None:
            x, y, w, h = bbox
            image = image[y:min(y + h, image.shape[0]), x:min(x + w, image.shape[1])]

        if self.transform is not None:
            image = self.transform(image)

        return image, instance_id, category_id, view_id

    def __len__(self):
        return len(self.dataset)


class Dataset(object):
    def __init__(self, data_dir, train_meta, test_meta):
        self.data_dir = data_dir

        train, num_train_instance_ids, num_train_category_ids, train_subpath_to_idx_dict = self._load_meta(data_dir, train_meta)
        self.train   = train
        self.num_train_ids   = num_train_instance_ids
        self.num_train_category_ids   = num_train_category_ids
        self.train_subpath_to_idx_dict   = train_subpath_to_idx_dict

        test, num_test_ids, _, test_subpath_to_idx_dict = self._load_meta(data_dir, test_meta)

        # list of tuples (image_path, identity_id, view_id, bbox)
        self.test   = test

        self.num_test_ids   = num_test_ids

        self.test_subpath_to_idx_dict   = test_subpath_to_idx_dict


    def _load_meta(self, dataset_dir, meta_file):
        datasets = []
        subpath_to_idx_dict = {}
        identity_set = set()
        category_to_idx_dict = {}
        with open(meta_file, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue

                image_id, label, view_id, image_subpath = line.strip().split()
                identity_id = int(label)

                bbox = None

                subpath_to_idx_dict[image_subpath] = len(datasets)
                
                category_str = image_subpath.split("/")[1]
                if category_str not in category_to_idx_dict:
                    category_to_idx_dict[category_str] = len(category_to_idx_dict)
                category_id = category_to_idx_dict[category_str]
                    
                datasets.append(
                    (os.path.join(dataset_dir, image_subpath), identity_id, category_id, int(view_id), bbox))

                identity_set.add(identity_id)

        return datasets, len(identity_set), len(category_to_idx_dict), subpath_to_idx_dict

    def print_stats(self):
        num_total_ids      = self.num_train_ids + self.num_test_ids

        num_train_images   = len(self.train)
        num_test_images   = len(self.test)
        num_total_images   = num_train_images + num_test_images

        print("##### Dataset Statistics #####")
        print("+----------------------------+")
        print("| Subset  | #IDs  | #Images  |")
        print("+----------------------------+")
        print("| Train   | {:5d} | {:8d} |".format(self.num_train_ids, num_train_images))
        print("| Test    | {:5d} | {:8d} |".format(self.num_test_ids, num_test_images))
        print("+----------------------------+")
        print("| Total   | {:5d} | {:8d} |".format(num_total_ids, num_total_images))
        print("+----------------------------+")
