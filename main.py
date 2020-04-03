'''
Copyright (c) 2020-present NAVER Corp.
MIT license
'''
# encoding: utf-8
# this code is modified from https://github.com/naver/cgd
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mxnet as mx
import numpy as np
import os
import sys
import random
import argparse

import dataset as D
import transforms as T
from model import Model
from loss import HPHNTripletLoss
from runner import Trainer, Evaluator
from util import SummaryWriter

# define argparse

parser = argparse.ArgumentParser(description='Embedding Expansion Official MXNet codes')
parser.add_argument('--gpu_idx', default=None, type=str,
                    help='gpu index')
parser.add_argument('--lr_decay_factor', default=0.5, type=float,
                    help='value for learning rate decay')
parser.add_argument('--epochs', default=5000, type=int,
                    help='total training epochs')
parser.add_argument('--save_dir', default='./log/will/be/saved/here', type=str,
                    help='path for train and eval log')
parser.add_argument('--base_lr_mult', default=1.0, type=float,
                    help='scale for gradients calculated at backbone')
parser.add_argument('--eval_epoch_term', default=50, type=int,
                    help='check every eval_epoch_term')
parser.add_argument('--beta', default=1.2, type=float,
                    help='beta is beta')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='base learning rate')
parser.add_argument('--optim', default='adam', type=str,
                    help='use adam')
parser.add_argument('--image_size', default=227, type=int,
                    help='width and height of input image')
parser.add_argument('--data_name', default='car196', type=str,
                    help='car196 | sop')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='start epoch')
parser.add_argument('--sigma', default=0.5, type=float,
                    help='sigma is sigma')
parser.add_argument('--data_dir', default='./data/CARS_196', type=str,
                    help='image_path')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum is momentum')
parser.add_argument('--summary_step', default=10, type=int,
                    help='write summary every summary_step')
parser.add_argument('--wd', default=0.0005, type=float,
                    help='scale for weight decay')
parser.add_argument('--embed_dim', default=512, type=int,
                    help='dimension of embeddings')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed value')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--soft_margin', default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='parameter for hphn triplet loss')
parser.add_argument('--lr_decay_epochs', default='10,20,40,80', type=str,
                    help='split by comma')
parser.add_argument('--n_inner_pts', default=2, type=int,
                    help='the number of inner points. when it is 0, no EE')
parser.add_argument('--ee_l2norm', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='whether do l2 normalizing augmented embeddings')
parser.add_argument('--alpha', default=10, type=float,
                    help='alpha is alpha')
parser.add_argument('--margin', default=1e-5, type=float,
                    help='margin')
parser.add_argument('--num_workers', default=10, type=int,
                    help='for data preprocessing')
parser.add_argument('--num_instances', default=32, type=int,
                    help='how many instances per class')
parser.add_argument('--backbone', default='googlenet', type=str,
                    help='googlenet')
parser.add_argument('--recallk', default='1,2,4,8', type=str,
                    help='k values for recall')
parser.add_argument('--loss', default='hphn-triplet', type=str,
                    help='hphn-triplet')
parser.add_argument('--kvstore', default='device', type=str,
                    help='kvstore')


def add_best_values_summary(summary_writer, global_step, epoch, recallk, best_recall):
    if summary_writer is None:
        return
    summary_writer.add_scalar('metric/R%d/best' % (recallk), best_recall, global_step)
    summary_writer.add_scalar('metric_epoch/R%d/best' % (recallk), best_recall, epoch)

def add_summary(summary_writer, step, epoch, ranks, recall_at_ranks):
    for recallk, recall in zip(ranks, recall_at_ranks):
        if summary_writer is not None:
            summary_writer.add_scalar('metric/R%d' % (recallk), recall, step)
            summary_writer.add_scalar('metric_epoch/R%d' % (recallk), recall, epoch)
        print("R@{:3d}: {:.4f}".format(recallk, recall))

def evaluate_and_log(summary_writer, evaluator, ranks, step, epoch, best_metrics):
    metrics = []

    distmat, labels = evaluator.get_distmat()
    recall_at_ranks = evaluator.get_metric_at_ranks(distmat, labels, ranks)

    add_summary(summary_writer, step, epoch, ranks, recall_at_ranks)

    metrics.append(recall_at_ranks[0])

    for idx, best_recall1 in enumerate(best_metrics):
        recall1 = metrics[idx]
        if recall1 > best_recall1:
            best_recall1 = recall1
            best_metrics[idx] = best_recall1

            add_best_values_summary(summary_writer, step, epoch if epoch is not None else None,
                                    ranks[0], best_recall1)

    return best_metrics


def main():
    args = parser.parse_args()
    
    # define args more
    args.train_meta = './meta/CARS196/train.txt'
    args.test_meta = './meta/CARS196/test.txt'
    
    args.lr_decay_epochs = [int(epoch) for epoch in args.lr_decay_epochs.split(',')]
    args.recallk = [int(k) for k in args.recallk.split(',')]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
    args.ctx = [mx.gpu(0)]

    print(args)
    
    # Set random seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load image transform
    train_transform, test_transform = T.get_transform(image_size=args.image_size)

    # Load data loader
    train_loader, test_loader = D.get_data_loader(args.data_dir, args.train_meta, args.test_meta, train_transform, test_transform,
                                                  args.batch_size, args.num_instances, args.num_workers)

    # Load model
    model = Model(args.embed_dim, args.ctx)
    model.hybridize()

    # Load loss
    loss = HPHNTripletLoss(margin=args.margin, soft_margin=False, num_instances=args.num_instances, n_inner_pts=args.n_inner_pts, l2_norm=args.ee_l2norm)

    # Load logger and saver
    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard_log'))

    print("steps in epoch:", args.lr_decay_epochs)
    steps = list(map(lambda x: x*len(train_loader) , args.lr_decay_epochs))
    print("steps in iter:", steps)
    lr_schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_decay_factor)
    lr_schedule.base_lr = args.lr

    # Load optimizer for training
    optimizer = mx.gluon.Trainer(model.collect_params(),
                                 'adam', {'learning_rate': args.lr, 'wd': args.wd},
                                 kvstore=args.kvstore)
    
    # Load trainer & evaluator
    trainer   = Trainer(model, loss, optimizer, train_loader, summary_writer, args.ctx,
                        summary_step=args.summary_step,
                        lr_schedule=lr_schedule)
    
    evaluator = Evaluator(model, test_loader, args.ctx)
        
    best_metrics = [0.0]  # all query

    global_step = args.start_epoch * len(train_loader)
    
    # Enter to training loop
    print("base lr mult:", args.base_lr_mult)
    for epoch in range(args.start_epoch, args.epochs):
        model.backbone.collect_params().setattr('lr_mult', args.base_lr_mult)
            
        trainer.train(epoch)
        global_step = (epoch + 1) * len(train_loader)
        if (epoch + 1) % args.eval_epoch_term == 0:
            old_best_metric = best_metrics[0]
            # evaluate_and_log(summary_writer, evaluator, ranks, step, epoch, best_metrics)
            best_metrics = evaluate_and_log(summary_writer, evaluator, args.recallk,
                                        global_step, epoch + 1,
                                        best_metrics=best_metrics)
            if best_metrics[0] != old_best_metric:
                save_path = os.path.join(args.save_dir, 'model_epoch_%05d.params' % (epoch + 1))
                model.save_parameters(save_path)
        sys.stdout.flush()
    

if __name__ == '__main__':
    # https://github.com/dmlc/gluon-cv/issues/493
    sys.setrecursionlimit(2000)

    main()
