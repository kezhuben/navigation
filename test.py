#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/9/24 23:28
# desc: test the mnist data.
# ======================================================

import tensorflow as tf
from nets.dense_nets import DenseNets
from nets_params.dense_nets_params import dense_nets_params
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
from PIL import Image

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
dense_net = DenseNets(mnist.train, dense_nets_params)
# dense_net.train()
dense_net.inference(mnist.test)



# x = [1,2,3]
# for i in x:
#     plt.plot(i, i, marker=(2, 0, 119.48), markersize=20, linestyle='None')
#
# plt.xlim([0,4])
# plt.ylim([0,4])
#
# plt.show()
#
#
# img = Image.open("C:\\Users\\yangcheng\\Desktop\\交叉路识别\\数据\\交叉路.png")
# img.show()
# # new_img = Image.new('RGB', (pixel, pixel), background_color)
# # do not tailor the rotated image(expand = True).
# rotate_img = img.rotate(45, expand=True)
# rotate_img.show()

