from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from collections import OrderedDict
import logging
import os
import shutil

from ..tf_unet_1 import util
from ..tf_unet_1.layers import (weight_variable, bias_variable,
                                conv2d, max_pool, avg_pool,
                                pixel_wise_softmax_2, cross_entropy)
from ..tf_unet_1 import unet


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def create_conv_net(x, keep_prob, channels, n_class, layers=7,
                    transfer_layers=3, features_root=16, filter_size=3,
                    pool_size=2, summaries=True):
    """
    Creates a new convolutional sparse unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param transfer_layers: number of layers to keep for transfer learning
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    nx_static = x.get_shape()[1].value
    x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]

    transfer_weights = []
    transfer_biases = []
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    dw_h_convs = OrderedDict()

    # down layers
    for layer in range(layers):
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable(
                [filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable(
                [filter_size, filter_size, features//2, features], stddev)

        w2 = weight_variable(
            [filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])

        conv1 = conv2d(in_node, w1, keep_prob)
        # conv1 = tf.Print(conv1, [tf.reduce_max(w1)], "max layer %d.0 weight:" % layer)
        # conv1 = tf.Print(conv1, [tf.reduce_max(b1)], "max layer %d.0 bias:" % layer)
        tmp_h_conv = tf.nn.elu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        # conv2 = tf.Print(conv2, [tf.reduce_max(w2)], "max layer %d.5 weight:" % layer)
        # conv2 = tf.Print(conv2, [tf.reduce_max(b2)], "max layer %d.5 bias:" % layer)
        dw_h_convs[layer] = tf.nn.elu(conv2 + b2)

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        if layer < transfer_layers:
            transfer_weights.append((w1, w2))
            transfer_biases.append((b1, b2))

        if layer < layers - 1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]

    pool_size = nx_static // 2**(layers - 1)
    in_node = avg_pool(dw_h_convs[layers - 1], pool_size)

    # fully connected layer
    x_vector = tf.reshape(in_node, tf.stack([batch_size, features]))
    stddev = tf.to_float(tf.sqrt(2 / features))
    weight = weight_variable([features, n_class], stddev)
    bias = bias_variable([n_class])
    output = tf.nn.bias_add(tf.matmul(x_vector, weight), bias)

    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.summary.image('summary_conv_%02d_01' % i,
                             unet.get_image_summary(c1))
            tf.summary.image('summary_conv_%02d_02' % i,
                             unet.get_image_summary(c2))

        for k in pools.keys():
            tf.summary.image(
                'summary_pool_%02d' % k, unet.get_image_summary(pools[k]))

        for k in dw_h_convs.keys():
            tf.summary.histogram(
                "dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)
    variables.append(weight)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)
    variables.append(bias)

    transfer_variables = []
    for w1, w2 in transfer_weights:
        transfer_variables.append(w1)
        transfer_variables.append(w2)

    for b1, b2 in transfer_biases:
        transfer_variables.append(b1)
        transfer_variables.append(b2)

    return output, variables, transfer_variables


class SparseUnet(object):
    """
    A convolutional network with early layers matching a Unet, which can be
    used for pretraining.

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param weight_decay: (optional) weight decay, defaults to 0.
    """
    def __init__(self, channels=1, n_class=2, weight_decay=0, **kwargs):
        tf.reset_default_graph()

        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None, 512, 512, channels])
        self.y = tf.placeholder("float", shape=[None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep probability

        logits, self.variables, transfer_variables = create_conv_net(
            self.x, self.keep_prob, channels, n_class, **kwargs)

        self.cost = self._get_cost(logits, weight_decay)

        self.gradients_node = tf.gradients(self.cost, self.variables)
        self.predicter = tf.nn.softmax(logits)
        self.cross_entropy = cross_entropy(self.y, self.predicter)
        self.correct_pred = tf.equal(
            tf.argmax(self.predicter, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver(var_list=self.variables)
        self.transfer_saver = tf.train.Saver(var_list=transfer_variables)

    def _get_cost(self, logits, weight_decay):
        y = self.y
        # logits = tf.Print(logits, [logits], "Logits: ", summarize=40)
        # y = tf.Print(y, [y], "Y: ", summarize=40)
        loss_map = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y)
        # loss_map = tf.Print(loss_map, [loss_map], "Loss map: ", summarize=40)
        loss = tf.reduce_mean(loss_map)

        if weight_decay > 0:
            regularizers = sum(
                [tf.nn.l2_loss(variable) for variable in self.variables])
            loss += weight_decay * regularizers

        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, labels]
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run(
                self.predicter,
                feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.}
            )

        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        save_path = self.saver.save(sess, model_path)
        return save_path

    def save_transfer(self, sess, model_path):
        """
        Saves transfer variables from the current session to a checkpoint
        :param sess: current session
        :param model_path: path to file system location
        """

        save_path = self.transfer_saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        self.saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


class Trainer(unet.Trainer):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.verification_batch_size = self.batch_size

    def _initialize(self, training_iters, output_path, restore_same_dir, var_list):
        global_step = tf.Variable(0)

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))

        if self.net.summaries:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step, var_list)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        do_transfer = self.transfer_dir is not None
        if do_transfer:
            transfer_path = os.path.abspath(self.transfer_dir)
        output_path = os.path.abspath(output_path)

        if not restore_same_dir:
            if do_transfer:
                logging.info("Removing '{:}'".format(transfer_path))
                shutil.rmtree(transfer_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if do_transfer and not os.path.exists(transfer_path):
            logging.info("Allocating '{:}'".format(transfer_path))
            os.makedirs(transfer_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, restore_path=None,
              transfer_path=None, training_iters=10, epochs=100, dropout=0.75,
              display_step=1, restore=False, write_graph=False, var_list=None):
        """
        Launches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param restore_path: if not None, path to restore model from; else equals output path
        :param transfer_path: if not None, path to save transfer variables only
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout keep probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param var_list: if not None, specifies which variables should be trained.
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path

        restore_same_dir = restore
        if restore_path is None and restore:
            restore_path = output_path
            restore_same_dir = False

        self.transfer_dir = transfer_path
        do_save_transfer = transfer_path is not None
        if do_save_transfer:
            transfer_path = os.path.join(transfer_path, "model.cpkt")

        init = self._initialize(training_iters, output_path, restore_same_dir,
                                var_list)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb",
                                     False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(restore_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            test_x, test_y = data_provider(self.verification_batch_size)
            self.make_prediction(sess, test_x, test_y)
            summary_writer = tf.summary.FileWriter(output_path,
                                                   graph=sess.graph)
            logging.info("Start optimization")

            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                for step in range(epoch * training_iters,
                                  (epoch + 1) * training_iters):
                    batch_x, batch_y = data_provider(self.batch_size)

                    # run optimization op (backprop)
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost,
                         self.learning_rate_node, self.net.gradients_node),
                        feed_dict={
                            self.net.x: batch_x,
                            self.net.y: batch_y,
                            self.net.keep_prob: dropout})

                    if avg_gradients is None:
                        avg_gradients = [
                            np.zeros_like(gradient) for gradient in gradients]
                    for i in range(len(gradients)):
                        avg_gradients[i] = (
                            avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (
                            gradients[i] / (step + 1))

                    norm_gradients = [
                        np.linalg.norm(gradient) for gradient in avg_gradients]
                    self.norm_gradients_node.assign(norm_gradients).eval()

                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step,
                                                    batch_x, batch_y)

                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.make_prediction(sess, test_x, test_y)

                save_path = self.net.save(sess, save_path)
                if do_save_transfer:
                    transfer_path = self.net.save_transfer(sess, transfer_path)

            logging.info("Optimization Finished!")
            return save_path, transfer_path

    def make_prediction(self, sess, batch_x, batch_y):
        """
        Like Unet's store_prediction, but don't save an image.
        """
        loss, accuracy = sess.run(
            (self.net.cost, self.net.accuracy), feed_dict={
                self.net.x: batch_x,
                self.net.y: batch_y,
                self.net.keep_prob: 1.})

        logging.info("Test: loss = {:.4f}, acc = {:.4f}".format(
            loss, accuracy))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x,
                               batch_y):
        # calculate batch loss and accuracy
        summary_str, loss, acc = sess.run(
            (self.summary_op, self.net.cost, self.net.accuracy),
            feed_dict={
                self.net.x: batch_x,
                self.net.y: batch_y,
                self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, loss = {:.4f}, acc = {:.4f}".format(
            step, loss, acc))
