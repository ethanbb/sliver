from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from ..tf_unet_1 import util
from ..tf_unet_1.layers import (weight_variable, bias_variable,
                                conv2d, max_pool, pixel_wise_softmax_2,
                                cross_entropy)


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
        tmp_h_conv = tf.nn.elu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        dw_h_convs[layer] = tf.nn.elu(conv2 + b2)

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        if layer < transfer_layers:
            transfer_weights.append((w1, w2))
            transfer_biases.append((b1, b2))

        pools[layer] = max_pool(dw_h_convs[layer], pool_size)
        in_node = pools[layer]

    # fully connected layer
    x_vector = tf.reshape(in_node, tf.stack([batch_size, -1]))
    features_in = tf.shape(x_vector)[1]
    stddev = np.sqrt(2 / features_in)
    weight = weight_variable([features_in, n_class], stddev)
    bias = bias_variable([n_class])
    output = tf.nn.bias_add(tf.matmul(x_vector, weight), bias)

    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
            tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))

        for k in pools.keys():
            tf.summary.image(
                'summary_pool_%02d' % k, get_image_summary(pools[k]))

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

        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep probability

        logits, self.variables, transfer_variables = create_conv_net(
            self.x, self.keep_prob, channels, n_class, **kwargs)

        self.cost = self._get_cost(logits, weight_decay)

        self.gradients_node = tf.gradients(self.cost, self.variables)
        self.predicter = tf.nn.softmax(logits)
        self.correct_pred = tf.equal(
            tf.argmax(self.predicter, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver(var_list=self.variables)
        self.transfer_saver = tf.train.Saver(var_list=transfer_variables)

    def _get_cost(self, logits, weight_decay):
        loss_map = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.y)
        base_loss = tf.reduce_mean(loss_map)

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
