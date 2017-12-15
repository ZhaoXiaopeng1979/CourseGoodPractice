#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf


class TextCNN(object):
    """A CNN for text classification.
    """
    def __init__(
        self, sequence_length, vocab_size, word_embed_size,
            num_classes, filter_num):

        # Placeholders for input, output
        self.input_x = tf.placeholder(
            tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.int32, shape=[None, ], name='input_y')

        # Embedding layer
        with tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, word_embed_size], -1.0, 1.0),
                name='W')
            self.embeds = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embeds_expanded = tf.expand_dims(self.embeds, -1)

        # Convolution + maxpool layer
        with tf.name_scope('conv-maxpool'):
            filter_num = 64
            window_size = 3
            filter_shape = [window_size, word_embed_size, 1, filter_num]
            W = tf.Variable(tf.random_uniform(filter_shape, -1.0, 1.0), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[filter_num]), name="b")
            conv = tf.nn.conv2d(
                self.embeds_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv')
            conv_hidden = tf.nn.tanh(tf.add(conv, b), name='tanh')
            pool = tf.nn.max_pool(
                conv_hidden,
                ksize=[1, sequence_length - window_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool')
            pool_shape = pool.get_shape().as_list()
            # pool_shape[0] 为一个 batch 中数据的个数，即评论条数
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            # 通过 tf.reshape 函数把 pool 层的输出编程一个 batch 的向量
            self.pool_flat = tf.reshape(pool, [-1, nodes])  # -1 表示尽可能的展平

        # Final scores and predictions
        with tf.name_scope('output'):
            self.logits = tf.layers.dense(self.pool_flat, num_classes)
            self.y = tf.nn.softmax(self.logits)

        # CalculateMean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope('accuracy'):
            # 计算预测值
            self.pred = tf.argmax(self.y, 1)
            # 判断两个张亮的每一维度是否相等
            correct_prediction = tf.equal(tf.cast(self.pred, tf.int32), self.input_y)
            # 先将布尔型的数值转为实数型，然后计算平均值
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
