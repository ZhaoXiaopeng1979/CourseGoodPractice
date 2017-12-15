# /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import click
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix as cm


from text_cnn import TextCNN
from text_helpers import build_dataset


class Parameters():
    """Parameters for command(clean, train, eval...)."""
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path


@click.group()
@click.option('--train-path', default='data/train_data.txt',
              help='Default: data/train_data.txt.')
@click.option('--test-path', default='data/test_data.txt',
              help='Default: data/test_data.txt.')
@click.pass_context
def cli(ctx, train_path, test_path):
    """CNN for Text Classification in Tensorflow.

    Examples:

    \b
        python cnn.py train  # train

    \b
        python cnn.py train --confusion-matrix  # plot confusion matrix

    \b
        python cnn.py --train-path train_shuffle.txt --test-path test_shuffle.txt clean  # text clean
    """
    ctx.obj = Parameters(train_path, test_path)


@cli.command()
@click.option('--stopwords-path', default='data/stop_words_chinese.txt')
@click.option('--sequence-length', default=20)
@click.option('-n', default=10,
              help='Find the common words that count is over n.')
@click.pass_obj
def clean(ctx, n, stopwords_path, sequence_length):
    print("Cleaning...")
    build_dataset(ctx.train_path, ctx.test_path,
                  stopwords_path, n, sequence_length)


@cli.command()
@click.option('--vocab-size', default=80000)
@click.option('--num-classes', default=2)
@click.option('--filter-num', default=64)
@click.option('--batch-size', default=50)
@click.option('--word-embed-size', default=128)
@click.option('--training-steps', default=10)
@click.option('--learning-rate', default=0.001)
@click.option('--print-loss-every', default=2)
@click.option('--confusion-matrix', is_flag=True)
@click.pass_obj
def train(ctx, vocab_size, num_classes, filter_num,
          batch_size, word_embed_size, training_steps,
          learning_rate, print_loss_every, confusion_matrix):

    # Load dataset
    train = np.loadtxt(ctx.train_path, dtype=int)
    test = np.loadtxt(ctx.test_path, dtype=int)
    x_train = train[:, :-1]
    y_train = train[:, -1:].reshape((-1,))
    x_test = test[:, :-1]
    y_test = test[:, -1:].reshape((-1,))
    sequence_length = x_train.shape[1]

    with tf.Graph().as_default():
        cnn = TextCNN(sequence_length, vocab_size, word_embed_size,
                      num_classes, filter_num)

        # Set feed_dict
        input_x, input_y = cnn.input_x, cnn.input_y
        train_feed_dict = {input_x: x_train, input_y: y_train}
        test_feed_dict = {input_x: x_test, input_y: y_test}

        # Train
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cnn.loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(training_steps):
                batch_data = train[np.random.randint(
                    train.shape[0], size=batch_size), :]
                X = batch_data[:, :-1]
                Y = batch_data[:, -1:].reshape((-1,))
                feed_dict = {input_x: X, input_y: Y}
                sess.run(train_step, feed_dict=feed_dict)
                if i % print_loss_every == 0:
                    total_cross_entropy = cnn.loss.eval(feed_dict=feed_dict)
                    train_accuracy = cnn.accuracy.eval(feed_dict=train_feed_dict)
                    test_accuracy = cnn.accuracy.eval(feed_dict=test_feed_dict)
                    test_pred = cnn.pred.eval(feed_dict=test_feed_dict)
                    print("After %d training steps, cross entropy on batch data is"
                          " %f, trian accuracy is %.2f, test accuracy is %.2f" % (
                              i, total_cross_entropy, train_accuracy, test_accuracy))

        if confusion_matrix:
            binary = cm(
                y_true=y_test, y_pred=test_pred
            )
            print('\n', 'Confusion Matrix: ')
            print(binary)
            plot_confusion_matrix(binary)
            plt.show()


if __name__ == '__main__':
    cli()
