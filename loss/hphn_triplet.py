'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np

import time
import datetime
import csv
import os

from .embedding_aug import get_embedding_aug

def euclidean_dist(x, y, clip_min=1e-12, clip_max=1e12):
    m, n = x.shape[0], y.shape[0]

    squared_x = torch.pow(x, 2).sum(axis=1, keepdims=True).broadcast_to((m, n))
    squared_y = torch.pow(y, 2).sum(axis=1, keepdims=True).broadcast_to((n, m)).T

    dist = squared_x + squared_y
    dist = dist - 2 * torch.dot(x, y.T)
    dist = dist.clip(a_min=clip_min, a_max=clip_max).sqrt()

    return dist


class HPHNTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2, soft_margin=False, weight=None, batch_axis=0, num_instances=2, n_inner_pts=0, l2_norm=True):
        super(HPHNTripletLoss, self).__init__()
        self.margin = margin
        self.soft_margin = soft_margin
        self.num_instance = num_instances
        self.n_inner_pts = n_inner_pts
        self.batch_size = None
        self.l2_norm = l2_norm

    def forward(self, embeddings, labels):
        total_start_time = time.time()
        gen_time = 0
        self.batch_size = embeddings.shape[0]
        if self.n_inner_pts != 0:
            gen_start_time = time.time()
            embeddings, labels = get_embedding_aug(embeddings, labels, self.num_instance, self.n_inner_pts, self.l2_norm)
            gen_time = time.time() - gen_start_time
        dist_mat = euclidean_dist(embeddings, embeddings)
        dist_ap, dist_an = self.hard_example_mining(dist_mat, labels)

        if self.soft_margin:
            loss = torch.log(1 + torch.exp(dist_ap - dist_an))
        else:
            loss = torch.relu(dist_ap - dist_an + self.margin)
        total_time = time.time() - total_start_time

        return loss

    def hard_example_mining(self, dist_mat, labels, return_inds=False):
        assert len(dist_mat.shape) == 2
        assert dist_mat.shape[0] == dist_mat.shape[1]

        N = dist_mat.shape[0]

        is_pos = torch.equal(labels.broadcast_to((N, N)), labels.broadcast_to((N, N)).T).astype('float32')
        is_neg = torch.not_equal(labels.broadcast_to((N, N)), labels.broadcast_to((N, N)).T).astype('float32')

        dist_pos = dist_mat * is_pos
        if self.n_inner_pts != 0:
            dist_ap = torch.max(dist_pos[:self.batch_size, :self.batch_size], axis=1)
        else:
            dist_ap = torch.max(dist_pos, axis=1)

        dist_neg = dist_mat * is_neg + torch.max(dist_mat, axis=1, keepdims=True) * is_pos
        dist_an = torch.min(dist_neg, axis=1)

        if self.n_inner_pts != 0:
            num_group = N // self.batch_size
            dist_an = torch.min(torch.reshape(dist_an, (num_group, self.batch_size)), axis=0) # include synthetic positives

        return dist_ap, dist_an
