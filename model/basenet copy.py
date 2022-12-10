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

from gluoncv import model_zoo
from gluoncv2.model_provider import get_model as gcv2_get_model, _models as gcv2_models

from pprint import pprint


class Backbone(mx.gluon.HybridBlock):
    def __init__(self, name, ctx):
        super(Backbone, self).__init__()

        name = name.lower()     #googlenet

        try:
            net = model_zoo.get_model(name, ctx=ctx, pretrained=True)
        except ValueError as e1:
            try:
                net = gcv2_get_model(name, ctx=ctx, pretrained=True)
            except ValueError:
                e2 = '%s' % ('\n\t'.join(sorted(gcv2_models.keys())))
                raise ValueError('{}\n\t{}'.format(e1, e2))

        with net.name_scope():
            self.base = mx.gluon.nn.HybridSequential('')
            self.base.add(net.conv1)
            self.base.add(net.maxpool1)
            self.base.add(net.conv2)
            self.base.add(net.conv3)
            self.base.add(net.maxpool2)
            self.base.add(net.inception3a)
            self.base.add(net.inception3b)
            self.base.add(net.maxpool3)
            self.base.add(net.inception4a)
            self.base.add(net.inception4b)
            self.base.add(net.inception4c)
            self.base.add(net.inception4d)
            self.base.add(net.inception4e)
            self.base.add(net.maxpool4)
            self.base.add(net.inception5a)
            self.base.add(net.inception5b)

    def hybrid_forward(self, F, x):
        return self.base(x)


class Model(mx.gluon.HybridBlock):
    def __init__(self, embed_dim, ctx):
        super(Model, self).__init__()

        self.embed_dim = embed_dim
        self.ctx = ctx
        
        self.backbone = Backbone('googlenet', ctx)
        
        with self.name_scope():
            self.embedding_layer = mx.gluon.nn.Dense(embed_dim, weight_initializer=mx.initializer.Xavier(magnitude=2))
            self.pooling_layer = mx.gluon.nn.GlobalAvgPool2D()
        
        self.embedding_layer.initialize(ctx=ctx)
        self.pooling_layer.initialize(ctx=ctx)

    def hybrid_forward(self, F, x, target_vectors=None):
        cnn_features = self.backbone(x)
        
        pooled_features = self.pooling_layer(cnn_features)

        embed = self.embedding_layer(pooled_features)
        
        final_embed = F.L2Normalization(embed, mode='instance')

        return final_embed

