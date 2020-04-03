'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mxnet as mx
import time


class Trainer(object):
    def __init__(self, model, loss, optimizer, data_loader, summary_writer, ctx,
                 summary_step,                 
                 lr_schedule):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.data_loader = data_loader
        self.ctx = ctx
        self.summary_step = summary_step
        self.lr_schedule = lr_schedule

    def _train_step(self, inputs):
        images, instance_ids, category_ids, _ = inputs
        batch_size = images.shape[0]
        data = mx.gluon.utils.split_and_load(images, self.ctx)
        instance_label = mx.gluon.utils.split_and_load(instance_ids, self.ctx)
        category_label = mx.gluon.utils.split_and_load(category_ids, self.ctx)
        
        loss_list = []
        with mx.autograd.record():
            for d, il, cl in zip(data, instance_label, category_label):
                labels = il
                embeddings = self.model(d)
                loss = self.loss(embeddings, labels)
                loss_list.append(loss)
        
        for loss in loss_list:
            loss.backward()
        self.optimizer.step(1)
        return loss_list

    def train(self, epoch):
        total_loss = 0
        epoch_start = time.time()

        for i, inputs in enumerate(self.data_loader):
            iter_count = len(self.data_loader)*epoch + i            
            self.optimizer.set_learning_rate(self.lr_schedule(iter_count))
            
            # Train step
            step_start = time.time()
            loss_list = self._train_step(inputs)
            loss_scalar = 0.0
            for loss in loss_list:
                loss_scalar += loss.mean().asscalar()

            step_end = time.time()

            global_step = epoch * len(self.data_loader) + i + 1

            # Logging every iteration
            if i % self.summary_step == 0:
                self.summary_writer.add_scalar('lr', self.optimizer.learning_rate, global_step)
                self.summary_writer.add_scalar('loss', loss_scalar, global_step)
            
            # Logging every epoch
            if i == len(self.data_loader) - 1:
                self.summary_writer.add_scalar('lr_epoch', self.optimizer.learning_rate, epoch + 1)
                self.summary_writer.add_scalar('loss_epoch', loss_scalar, epoch + 1)

            total_loss += loss_scalar

            print('[Epoch {}] Batch ({:03d}/{})\t'
                  'Batch Time: {:.3f}s\t'
                  'Global Step: {}\t'
                  'Loss: {:.3f}\t'
                  .format(epoch + 1, i + 1, len(self.data_loader), (step_end - step_start), global_step, loss_scalar))

        epoch_end = time.time()

        print('[Epoch {}] Finished\t'
              'Epoch Time: {:.3f}s\t'
              'Loss: {:.3f}\t'
              'Lr: {:.2e}\n'
              .format(epoch + 1, (epoch_end - epoch_start), total_loss, self.optimizer.learning_rate))
