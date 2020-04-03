'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tqdm import tqdm

import mxnet as mx
import numpy as np


class Evaluator(object):
    def __init__(self, model, test_loader, ctx):
        self.model = model
        self.test_loader = test_loader
        self.ctx = ctx

    def _eval_step(self, inputs):
        images, instance_ids, category_ids, view_ids = inputs
        data = mx.gluon.utils.split_and_load(images, self.ctx, even_split=False)
        instance_ids = instance_ids.asnumpy()
        view_ids = view_ids.asnumpy()
        feats = []
        for d in data:
            feats.append(self.model(d))
        feats = mx.nd.concatenate(feats, axis=0)
        return feats, instance_ids, view_ids

    
    def get_distmat(self):
        print('Extracting eval features...')
        features, labels = [], []
        for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            feature, instance_ids, view_ids = self._eval_step(inputs)
            features.append(feature.asnumpy())
            labels.extend(instance_ids)
        features = np.concatenate(features)
        labels = np.asarray(labels)
        
        m = features.shape[0]
        squared_sum_features = np.sum(features ** 2.0, axis=1, keepdims=True)
        distmat = squared_sum_features + squared_sum_features.transpose() - (2.0 * np.dot(features, features.transpose()))

        return distmat, labels


    def get_metric_at_ranks(self, distmat, labels, ranks):
        np.fill_diagonal(distmat, 100000.0)

        recall_at_ranks = []

        recall_dict = {k: 0 for k in ranks}

        max_k = np.max(ranks)

        # do partition
        arange_idx = np.arange(len(distmat))[:,None]
        part_idx = np.argpartition(distmat, max_k, axis=1)[:,:max_k]
        part_mat = distmat[arange_idx, part_idx]

        # do sort
        sorted_idx = np.argsort(part_mat, axis=1)#[::-1]
        top_k_idx = part_idx[arange_idx, sorted_idx]

        for top_k, gt in zip(top_k_idx, labels):
            top_k_labels = labels[top_k]
            for r in ranks:
                if gt in top_k_labels[:r]:
                    recall_dict[r] += 1

        for r in ranks:
            recall_at_ranks.append(recall_dict[r] / len(distmat))

        return recall_at_ranks
