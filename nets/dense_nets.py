#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/9/21 18:45
# desc: The Densely Connected Convolutional Networks. See: https://arxiv.org/pdf/1608.06993.pdf
# ======================================================

import tensorflow as tf


class DenseNets(object):
    def __init__(self, images_and_labels, params):
        """three input for dense net class.

        Args:
            images_and_labels: `iterator`, images data for training.
                                and labels , e.g. [[0,0,0,1,0,0,..],[...],...] images labels for training.
            params: `dict`, params details see: ../nets_params/dense_nets_params.py
        """
        self.images_and_labels_for_train = images_and_labels

        # params, details see: ../nets_params/dense_nets_params.py
        self.pixel = params["PIXEL"]
        self.channel = params["CHANNEL"]
        self.classes = params["CLASSES"]
        self.growth_rate = params['GROWTH_RATE']
        self.blocks = params['BLOCKS']
        self.layers_per_block = params['LAYERS_PER_BLOCK']
        self.compression_rate = params['COMPRESSION_RATE']
        self.learning_rate = params['LEARNING_RATE']
        self.keep_probability = params['KEEP_PROBABILITY']
        self.batch_size = params['BATCH_SIZE']
        self.epoch = params['EPOCH']

    def _conv_weight(self, shape):
        """method for convolution weight. """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32))

    def _conv_bias(self, shape):
        """method for convolution bias. """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32))

    def _batch_normalization(self, input_tensor, is_training):
        """batch normalization for dense nets.

        Args:
            input_tensor: `tensor`, the input tensor which needed normalized.
            is_training: `bool`, if true than update the mean/variance using moving average,
                                 else using the store mean/variance.

        Returns:
            normalized params.
        """
        return tf.layers.batch_normalization(input_tensor, training=is_training)

    def _add_dense_block(self,
                         input_tensor,
                         layers_per_block,
                         input_size,
                         output_size,
                         keep_probability,
                         is_training):
        """method for dense block before convolution.

        Args:
            input_tensor: `tensor`, the output tensor of previous layer.
            layers_per_block: `int`, the number of layers per dense block.
            input_size: `int`, input tensor chanel.
            output_size: `int`, output features size.
            keep_probability: `float`, keep probability for dropout.
            is_training: `bool`, if true than update the mean/variance using moving average,
                                 else using the store mean/variance.

        Returns:
            Dense block tensor.
        """
        # block_result = []
        for layer in range(layers_per_block):
            block_kernel_1x1 = self._conv_weight(
                [1, 1, int(input_size + layer * output_size), int(input_size + layer * output_size)])
            block_kernel_3x3 = self._conv_weight([3, 3, int(input_size + layer * output_size), output_size])

            # BN -> RELU -> conv 1x1 -> dropout
            block_result = self._batch_normalization(input_tensor, is_training)
            block_result = tf.nn.relu(block_result)
            block_result = tf.nn.conv2d(block_result, block_kernel_1x1, [1, 1, 1, 1], padding='VALID')
            block_result = tf.nn.dropout(block_result, keep_probability)

            # BN -> RELU -> conv 3x3 -> dropout
            block_result = self._batch_normalization(block_result, is_training)
            block_result = tf.nn.relu(block_result)
            block_result = tf.nn.conv2d(block_result, block_kernel_3x3, [1, 1, 1, 1], padding='SAME')
            block_result = tf.nn.dropout(block_result, keep_probability)

            # concat the tensor.
            input_tensor = tf.concat([input_tensor, block_result], axis=3)

        return input_tensor

    def _add_transition_layer(self, input_tensor, compression_rate, is_training):
        """transition layer between every two block.

        Args:
            input_tensor: `tensor`, the output tensor of previous layer.
            compression_rate: `float`, compression rate for reduce the output features from transition layer.
            is_training: `bool`, if true than update the mean/variance using moving average,
                                 else using the store mean/variance.

        Returns:
            tensor that after convolution and pool.
        """
        # input/output size for transition.
        input_size = int(input_tensor.get_shape()[-1])
        output_size = int(compression_rate * input_size)

        # weight for convolution 3x3.
        transition_kernel_3x3 = self._conv_weight([3, 3, input_size, output_size])
        # batch norm, relu, 3x3 with zero-padding convolution and 2x2 average pool.
        transition_result = self._batch_normalization(input_tensor, is_training)
        transition_result = tf.nn.relu(transition_result)
        transition_result = tf.nn.conv2d(transition_result, transition_kernel_3x3, [1, 1, 1, 1], padding='SAME')
        transition_result = tf.nn.max_pool(transition_result, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        return transition_result

    def _build_graph(self):
        """build the tensor flow graph and training.

        Returns:
            dense net graph.
        """

        # placeholder for batch input images(mnist is 28*28 pixel) and labels(mnist is 10 classes, one hot type).
        images = tf.placeholder(tf.float32, [None, self.pixel * self.pixel * self.channel], name='images')
        labels = tf.placeholder(tf.float32, [None, self.classes], name='labels')
        is_training = tf.placeholder(tf.bool)

        # build the graph.
        # reshape the images.
        images_reshape = tf.reshape(images, [-1, self.pixel, self.pixel, self.channel])
        # first convolution(3x3 kernel with zero padding).
        input_size = int(images_reshape.get_shape()[-1])
        first_kernel_1x1 = self._conv_weight([2, 2, input_size, int(self.growth_rate * 4)])
        images_reshape = self._batch_normalization(images_reshape, is_training)
        result = tf.nn.conv2d(images_reshape, first_kernel_1x1, [1, 1, 1, 1], padding='SAME')
        result = tf.nn.relu(result)
        result = tf.nn.max_pool(result, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # dense block with transition layer.
        for block in range(self.blocks):
            inner_input_size = int(result.get_shape()[-1])
            result = self._add_dense_block(result,
                                           self.layers_per_block,
                                           inner_input_size,
                                           self.growth_rate,
                                           self.keep_probability,
                                           is_training)
            result = self._add_transition_layer(result, self.compression_rate, is_training)

        # first full connected layer([batch, weight * height * channel] x [weight * height * channel, 2048] + [2048]).
        result_shape = result.get_shape()
        full_connected_weight1 = self._conv_weight(
            [int(result_shape[1]) * int(result_shape[2]) * int(result_shape[3]), 2048])
        full_connected_bias1 = self._conv_weight([2048])
        result = self._batch_normalization(result, is_training)
        result = tf.nn.relu(result)
        result_reshape = tf.reshape(result, [-1, int(result_shape[1]) * int(result_shape[2]) * int(result_shape[3])])
        result = tf.matmul(result_reshape, full_connected_weight1) + full_connected_bias1

        # second full connected layer([batch, 2048] x [2048, classes] + [classer]).
        full_connected_weight2 = self._conv_weight([2048, self.classes])
        full_connected_bias2 = self._conv_weight([self.classes])
        result = self._batch_normalization(result, is_training)
        result = tf.nn.relu(result)
        logits = tf.matmul(result, full_connected_weight2) + full_connected_bias2
        prediction = tf.nn.softmax(logits, name='prediction')

        # calculate the loss under cross entropy.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # loss = cross_entropy + tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * 0.01
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return images, labels, is_training, train_step, cross_entropy, prediction, accuracy

    def train(self):
        """train the dense net model.

        Returns:
            trained model.
        """
        # build the dense net graph.
        images, labels, is_training, train_step, cross_entropy, _, accuracy = self._build_graph()
        # training.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # loop in all epoch.
        for i in range(self.epoch):
            # loop all the data in one epoch.
            for j in range(40000 // self.batch_size):
                batch = self.images_and_labels_for_train.next_batch(self.batch_size)
                # feed the data in the placeholder.
                feed_dict = {
                    images: batch[0],
                    labels: batch[1],
                    is_training: True
                }
                train_step.run(feed_dict=feed_dict, session=sess)
                loss_cross_entropy, acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
                print("Training in epoch {} and step {}, the cross entropy is {} and accuracy is {}".format(i,
                                                                                                            j,
                                                                                                            loss_cross_entropy,
                                                                                                            acc))
        saver = tf.train.Saver(var_list=tf.global_variables())
        savepath = saver.save(sess, './models/dense_nets_model/dense_nets.ckpt')
        print("Model saving path is %s" % savepath)

    def inference(self, images_for_predict):
        """load the pre-trained model and do the inference.

        Args:
            images_for_predict: `tensor`, images for predict using the pre-trained model.

        Returns:
            the predict labels.
        """

        tf.reset_default_graph()
        images, labels, is_training, _, _, prediction, accuracy = self._build_graph()

        predictions = []
        correct = 0
        with tf.Session() as sess:
            # sess.run(tf.global_variables())
            # saver = tf.train.import_meta_graph('./models/dense_nets_model/dense_nets.ckpt.meta')
            # saver.restore(sess, tf.train.latest_checkpoint('./models/dense_nets_model/'))
            saver = tf.train.Saver()
            saver.restore(sess, './models/dense_nets_model/dense_nets.ckpt')
            for i in range(2000):
                pred, corr = sess.run([tf.argmax(prediction, 1), accuracy],
                                      feed_dict={
                                          images: [images_for_predict.images[i]],
                                          labels: [images_for_predict.labels[i]],
                                          is_training: False
                                      })
                correct += corr
                predictions.append(pred[0])
        print("PREDICTIONS:", predictions)
        print("ACCURACY:", correct / 2000)
