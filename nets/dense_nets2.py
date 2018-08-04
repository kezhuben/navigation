#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/10/15 11:21
# desc:
# ======================================================

import tensorflow as tf
from tensorpack import *


class DenseNets(object):
    def __init__(self, images, labels, params):
        """three input for dense net class.

        Args:
            images: `tensor`, images data for training.
            labels: `tensor`, e.g. [[0,0,0,1,0,0,..],[...],...] images labels for training.
            params: `dict`, params details see: ../nets_params/dense_nets_params.py
        """
        self.all_images = images
        self.all_labels = labels
        self.params = params

    def train(self, pixel, classes):
        growth_rate = self.params['GROWTH_RATE']
        blocks = self.params['BLOCKS']
        layers_per_block = self.params['LAYERS_PER_BLOCK']
        compression_rate = self.params['COMPRESSION_RATE']
        learning_rate = self.params['LEARNING_RATE']
        decay = self.params['DECAY']
        keep_probability = self.params['KEEP_PROBABILITY']
        batch_norm_epsilon = self.params['BATCH_NORM_EPSILON']
        epoch = self.params['EPOCH']
        batch_size = self.params['BATCH_SIZE']

        def _batch_normalization(input_tensor, is_training, batch_norm_epsilon, decay=0.999):
            """batch normalization for dense nets.

            Args:
                input_tensor: `tensor`, the input tensor which needed normalized.
                is_training: `bool`, if true than update the mean/variance using moving average,
                                     else using the store mean/variance.
                batch_norm_epsilon: `float`, param for batch normalization.
                decay: `float`, param for update move average, default is 0.999.

            Returns:
                normalized params.
            """
            # actually batch normalization is according to the channels dimension.
            input_shape_channels = int(input_tensor.get_shape()[-1])

            # scala and beta using in the the formula like that: scala * (x - E(x))/sqrt(var(x)) + beta
            scale = tf.Variable(tf.ones([input_shape_channels]))
            beta = tf.Variable(tf.zeros([input_shape_channels]))

            # global mean and var are the mean and var that after moving averaged.
            global_mean = tf.Variable(tf.zeros([input_shape_channels]), trainable=False)
            global_var = tf.Variable(tf.ones([input_shape_channels]), trainable=False)

            # if training, then update the mean and var, else using the trained mean/var directly.
            if is_training:
                # batch norm in the channel axis.
                axis = list(range(len(input_tensor.get_shape()) - 1))
                batch_mean, batch_var = tf.nn.moments(input_tensor, axes=axis)

                # update the mean and var.
                train_mean = tf.assign(global_mean, global_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(global_var, global_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(input_tensor,
                                                     batch_mean, batch_var, beta, scale, batch_norm_epsilon)
            else:
                return tf.nn.batch_normalization(input_tensor,
                                                 global_mean, global_var, beta, scale, batch_norm_epsilon)

        images = tf.placeholder(tf.float32, [batch_size, pixel * pixel])
        labels = tf.placeholder(tf.float32, [batch_size, classes])

        images_reshape = tf.reshape(images, [batch_size, pixel, pixel, 1])
        conv_weight1 = tf.Variable(tf.truncated_normal([2, 2, 1, 50]), name='weight1')
        images_reshape = _batch_normalization(images_reshape, True, batch_norm_epsilon)
        conv1 = tf.nn.conv2d(images_reshape, conv_weight1, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # dense block
        for i in range(layers_per_block):
            input_shape = pool1.get_shape()
            block_layer_weight0 = tf.Variable(tf.truncated_normal([1,1,int(input_shape[-1]),int(input_shape[-1])]),name='layer{}_weight0'.format(i))
            block_layer_weight = tf.Variable(tf.truncated_normal([3, 3, int(input_shape[-1]), 50]),
                                             name='layer{}_weight'.format(i))
            temp = _batch_normalization(pool1, True, batch_norm_epsilon)
            temp = tf.nn.relu(temp)
            temp = tf.nn.conv2d(temp, block_layer_weight0, [1, 1, 1, 1], padding='VALID')
            temp = tf.nn.dropout(temp, keep_probability)

            temp = _batch_normalization(temp, True, batch_norm_epsilon)
            temp = tf.nn.relu(temp)
            temp = tf.nn.conv2d(temp, block_layer_weight, [1, 1, 1, 1], padding='SAME')
            pool1 = tf.concat([temp, pool1], axis=3)

        conv_weight2 = tf.Variable(tf.truncated_normal([2, 2, 250, 500]), name='weight2')
        pool1 = _batch_normalization(pool1, True, batch_norm_epsilon)
        conv2 = tf.nn.relu(pool1)
        conv2 = tf.nn.conv2d(conv2, conv_weight2, [1, 1, 1, 1], padding='SAME')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # dense block
        for i in range(layers_per_block):
            input_shape = pool2.get_shape()
            block_layer_weight0 = tf.Variable(tf.truncated_normal([1, 1, int(input_shape[-1]), int(input_shape[-1])]),
                                              name='layer{}_weight0'.format(i))
            block_layer_weight = tf.Variable(tf.truncated_normal([3, 3, int(input_shape[-1]), 500]),
                                             name='layer{}_weight'.format(i))

            temp = _batch_normalization(pool2, True, batch_norm_epsilon)
            temp = tf.nn.relu(temp)
            temp = tf.nn.conv2d(temp, block_layer_weight0, [1, 1, 1, 1], padding='VALID')
            temp = tf.nn.dropout(temp, keep_probability)

            temp = _batch_normalization(temp, True, batch_norm_epsilon)
            temp = tf.nn.relu(temp)
            temp = tf.nn.conv2d(temp, block_layer_weight, [1, 1, 1, 1], padding='SAME')
            pool2 = tf.concat([temp, pool2], axis=3)

        conv_weight3 = tf.Variable(tf.truncated_normal([2, 2, 2500, 500]), name='weight3')
        pool2 = _batch_normalization(pool2, True, batch_norm_epsilon)
        conv3 = tf.nn.relu(pool2)
        conv3 = tf.nn.conv2d(conv3, conv_weight3, [1, 1, 1, 1], padding='SAME')
        pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')



        fc_weight = tf.Variable(tf.truncated_normal([4 * 4 * 500, 1000]), name='fc_weight')
        fc_bias = tf.Variable(tf.truncated_normal([1000]), name='fc_bias')
        pool3_reshape = tf.reshape(pool3, [-1, 4 * 4 * 500])
        pool3_reshape = _batch_normalization(pool3_reshape, True, batch_norm_epsilon)
        fc1 = tf.matmul(pool3_reshape, fc_weight) + fc_bias
        fc1 = tf.nn.relu(fc1)
        # fc1 = tf.nn.dropout(fc1,0.9)

        fc_weight2 = tf.Variable(tf.truncated_normal([1000, 10]), name='fc_weight2')
        fc_bias2 = tf.Variable(tf.truncated_normal([10]), name='fc_bias2')
        logit = tf.matmul(fc1, fc_weight2) + fc_bias2
        prediction = tf.nn.softmax(logit)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logit)
        train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # images_reshape = tf.reshape(images, [batch_size, pixel, pixel, 1])
        # input_size = int(images_reshape.get_shape()[-1])
        # first_kernel_3x3 = tf.Variable(tf.truncated_normal([3, 3, input_size, int(growth_rate * 4)], stddev=0.01,mean=0),name='firstW')
        # result = tf.nn.conv2d(images_reshape, first_kernel_3x3, [1, 1, 1, 1], padding='SAME')

        # result_temp1 = []
        # with tf.variable_scope('block1'):
        #
        #     for i in range(layers_per_block):
        #         input_shape_blokc1 = result.get_shape()
        #         # block_kernel_1x1 = tf.Variable(
        #         #     tf.truncated_normal([1, 1, int(input_shape_blokc1[-1]),
        #         #                          int(input_shape_blokc1[-1])],stddev=0.1,mean=0),
        #         #     name="block1_layer{}_weight1x1".format(i))
        #         block_kernel_3x3 = tf.Variable(
        #             tf.truncated_normal([1, 1, int(input_shape_blokc1[-1]), growth_rate], stddev=0.01,mean=0),
        #             name="block1_layer{}_weight3x3".format(i))
        #         result_temp1 = tf.nn.relu(result)
        #         # result_temp1 = tf.nn.conv2d(result_temp1, block_kernel_1x1, [1, 1, 1, 1], padding='VALID')
        #         # result_temp1 = tf.nn.dropout(result_temp1, keep_probability)
        #         # result_temp1 = tf.nn.relu(result_temp1)
        #         result_temp1 = tf.nn.conv2d(result_temp1, block_kernel_3x3, [1, 1, 1, 1], padding='SAME')
        #         result_temp1 = tf.nn.dropout(result_temp1, keep_probability)
        #         result = tf.concat([result, result_temp1], axis=3)

        # input_size_transition1 = int(result.get_shape()[-1])
        # output_size_transition1 = int(compression_rate * input_size_transition1)
        #
        # # weight for convolution 3x3.
        # transition1_kernel_1x1 = tf.Variable(
        #     tf.truncated_normal([1, 1, input_size_transition1, output_size_transition1], stddev=0.01,mean=0),
        #     name="trainsition1_weight")
        # # batch norm, relu, 3x3 with zero-padding convolution and 2x2 average pool.
        # transition1_result = tf.nn.relu(result)
        # transition1_result = tf.nn.conv2d(transition1_result, transition1_kernel_1x1, [1, 1, 1, 1], padding='VALID')
        # transition1_result = tf.nn.max_pool(transition1_result, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # result_temp2 = []
        # with tf.variable_scope('block2'):
        #     for i in range(layers_per_block):
        #         input_shape_blokc2 = transition1_result.get_shape()
        #
        #         # block_kernel_1x1 = tf.Variable(
        #         #     tf.truncated_normal([1, 1, int(input_shape_blokc2[-1]),
        #         #                          int(input_shape_blokc2[-1])],stddev=0.1),
        #         #     name="block2_layer{}_weight1x1".format(i))
        #         block_kernel_3x3 = tf.Variable(
        #             tf.truncated_normal([1, 1, int(input_shape_blokc2[-1]), growth_rate], stddev=0.01,mean=0),
        #             name="block2_layer{}_weight3x3".format(i))
        #         result_temp2 = tf.nn.relu(transition1_result)
        #         # result_temp2 = tf.nn.conv2d(result_temp2, block_kernel_1x1, [1, 1, 1, 1], padding='VALID')
        #         # result_temp2 = tf.nn.dropout(result_temp2, keep_probability)
        #         # result_temp2 = tf.nn.relu(result_temp2)
        #         result_temp2 = tf.nn.conv2d(result_temp2, block_kernel_3x3, [1, 1, 1, 1], padding='SAME')
        #         result_temp2 = tf.nn.dropout(result_temp2, keep_probability)
        #         transition1_result = tf.concat([transition1_result, result_temp2], axis=3)
        #
        # input_size_transition2 = int(result_temp2.get_shape()[-1])
        # output_size_transition2 = int(compression_rate * input_size_transition2)
        #
        # # weight for convolution 3x3.
        # transition2_kernel_1x1 = tf.Variable(
        #     tf.truncated_normal([1, 1, input_size_transition2, output_size_transition2], stddev=0.01),
        #     name="trainsition2_weight")
        # # batch norm, relu, 3x3 with zero-padding convolution and 2x2 average pool.
        # transition2_result = tf.nn.relu(result_temp2)
        # transition2_result = tf.nn.conv2d(transition2_result, transition2_kernel_1x1, [1, 1, 1, 1], padding='VALID')
        # transition2_result = tf.nn.max_pool(transition2_result, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        #
        # result_temp3 = []
        # with tf.variable_scope('block3'):
        #     for i in range(layers_per_block):
        #         input_shape_blokc3 = transition2_result.get_shape()
        #
        #         # block_kernel_1x1 = tf.Variable(
        #         #     tf.truncated_normal([1, 1, int(input_shape_blokc3[-1]),
        #         #                          int(input_shape_blokc3[-1])],stddev=0.1),
        #         #     name="block3_layer{}_weight1x1".format(i))
        #         block_kernel_3x3 = tf.Variable(
        #             tf.truncated_normal([1, 1, int(input_shape_blokc3[-1]), growth_rate], stddev=0.01,mean=0),
        #             name="block3_layer{}_weight3x3".format(i))
        #         result_temp3 = tf.nn.relu(transition2_result)
        #         # result_temp3 = tf.nn.conv2d(result_temp3, block_kernel_1x1, [1, 1, 1, 1], padding='VALID')
        #         # result_temp3 = tf.nn.dropout(result_temp3, keep_probability)
        #         # result_temp3 = tf.nn.relu(result_temp3)
        #         result_temp3 = tf.nn.conv2d(result_temp3, block_kernel_3x3, [1, 1, 1, 1], padding='SAME')
        #         result_temp3 = tf.nn.dropout(result_temp3, keep_probability)
        #         transition2_result = tf.concat([transition2_result, result_temp3], axis=3)
        #
        # input_size_transition3 = int(result.get_shape()[-1])
        # output_size_transition3 = int(compression_rate * input_size_transition3)
        #
        # # weight for convolution 3x3.
        # transition3_kernel_1x1 = tf.Variable(
        #     tf.truncated_normal([1, 1, input_size_transition3, output_size_transition3], stddev=0.01,mean=0),
        #     name="trainsition3_weight")
        # # batch norm, relu, 3x3 with zero-padding convolution and 2x2 average pool.
        # transition3_result = tf.nn.relu(result)
        # transition3_result = tf.nn.conv2d(transition3_result, transition3_kernel_1x1, [1, 1, 1, 1], padding='VALID')
        # transition3_result = tf.nn.max_pool(transition3_result, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # transition3_result_shape = result.get_shape()
        # # result = self._batch_normalization(result, True, batch_norm_epsilon)
        # transition3_result = tf.nn.relu(result)
        # transition3_result = tf.nn.max_pool(transition3_result,
        #                                     [1, int(transition3_result_shape[-2]), int(transition3_result_shape[-2]),
        #                                      1],
        #                                     [1, int(transition3_result_shape[-2]), int(transition3_result_shape[-2]),
        #                                      1],
        #                                     padding='VALID')
        # final_result = tf.reshape(transition3_result, [-1, int(transition3_result_shape[-1])])
        # linear_weight = tf.Variable(tf.truncated_normal([int(transition3_result_shape[-1]), classes], stddev=.01,mean=0),
        #                             name='linear_weight')
        # linear_bias = tf.constant(0.1,shape=[classes])
        # logits = tf.matmul(final_result, linear_weight) + linear_bias
        # prediction = tf.nn.softmax(logits)
        #
        # # calculate the loss under cross entropy.
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # loss = cross_entropy + tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * 0.01
        # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        # correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # loop in all epoch.
        for i in range(epoch):
            # loop all the data in one epoch.
            for j in range(40000 // batch_size):
                batch = self.all_labels.next_batch(batch_size)
                # feed the data in the placeholder.
                feed_dict = {
                    images: batch[0],
                    labels: batch[1]
                }
                train_step.run(feed_dict=feed_dict, session=sess)
                train_loss = cross_entropy.eval(feed_dict=feed_dict, session=sess)
                train_accuracy = accuracy.eval(feed_dict=feed_dict, session=sess)
                print "step {}, training accuracy {}".format(j, train_accuracy)
