#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/10/3 18:07
# desc: The default parameters for dense nets, algorithm details see: ../nets/dense_nets.py
# ======================================================

dense_nets_params = {
    # the input images pixel.
    "PIXEL": 28,

    # the input images channel.
    "CHANNEL": 1,

    # the number of classes of the all input images.
    "CLASSES": 10,

    # growth rate for dense block layers, params from the dense nets paper, default is 12.
    # it means that every layer in the block will output `GROWTH_RATE` features.
    "GROWTH_RATE": 20,

    # blocks for the dense nets, params from the dense nets paper, default is 3.
    # the default dense nets structure is:
    #     input -> transition layer -> block -> transition layer -> block -> transition layer
    #     -> block -> transition layer -> output
    "BLOCKS": 3,

    # the number of layers per block, params from the dense net paper, default is 4.
    # the default dense nets block structure is:
    #     transition layer -> batch normalize -> RELU -> convolution 1x1(no padding) -> batch normalize
    #     -> RELU -> convolution 3x3(padding)
    "LAYERS_PER_BLOCK": 4,

    # compression rate for reduce the output features from transition layer, params from the dense net paper,
    # default is 0.5.
    # it means that when the transition layer output k features, we reduce it to `COMPRESSION_RATE` * k.
    "COMPRESSION_RATE": 2,

    # learning rate for gradient decent training.
    "LEARNING_RATE": 0.01,

    # keep probability for drop out.
    "KEEP_PROBABILITY": 0.8,

    # epoch, total number of training.
    "EPOCH": 3,

    # batch size of data for training.
    "BATCH_SIZE": 100
}
