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
import tensorflow as tf


class SummaryWriter(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        try:
            self.tf2 = False
            self.summary_writer = tf.summary.FileWriter(log_dir)
        except:
            self.tf2 = True
            self.summary_writer = tf.summary.create_file_writer(log_dir)
            

    def add_scalar(self, tag, value, step):
        if self.tf2:
            with self.summary_writer.as_default():
                tf.summary.scalar(tag, value, step)
        else:
            # tensorflow 1.x tensorboard
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=value)
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()
