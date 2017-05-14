from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import logging
import numpy as np
from tf_unet_1 import unet
import conv_rnn_cell as crc
import conv_rnn as cr
from tensorflow.python.ops import variable_scope as vs
from tf_unet_1.layers import (weight_variable, weight_variable_devonc,
                              bias_variable, conv2d, deconv2d, max_pool,
                              crop_and_concat, pixel_wise_softmax_2,
                              cross_entropy)

import tensorflow as tf


class RUnet(unet.Unet):
    """Model that incorporates a Unet and a bidirectional convolutional LSTM
    n_lstm_layers is the number of vertical bidirectional LSTM layers
    All other inputs are the same as the inputs to Unet.__init__.
    """
    def __init__(self, batch_size, n_lstm_layers=1, channels=3, n_class=2,
                 cost="cross_entropy", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[batch_size, 512, 512, channels])
        self.y = tf.placeholder("float", shape=[batch_size, 512, 512, n_class])

        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep probability

        # set up Unet
        feature_maps, self._unet_logits, self.unet_variables, self.offset = create_conv_net(
            self.x, self.keep_prob, channels, n_class, **kwargs)

        self.cost_fn = cost
        self.cost_kwargs = cost_kwargs

        #  batch dimension of feature_maps becomes time points in LSTM
        lstm_input = tf.unstack(feature_maps)
        lstm_input = [tf.expand_dims(x, 0) for x in lstm_input]
        self._lstm_logits, self.lstm_variables = create_lstm(
            lstm_input, n_class, n_lstm_layers, **kwargs)

        # This property is meant to be settable
        self.use_lstm = True

    @property
    def _logits(self):
        return self._lstm_logits if self.use_lstm else self._unet_logits

    @property
    def variables(self):
        if self.use_lstm:
            return self.lstm_variables + self.unet_variables
        else:
            return self.unet_variables

    @property
    def cost(self):
        return self._get_cost(self._logits, self.cost_fn, self.cost_kwargs)

    @property
    def gradients_node(self):
        return tf.gradients(self.cost, self.variables)

    @property
    def cross_entropy(self):
        return tf.reduce_mean(cross_entropy(
            tf.reshape(self.y, [-1, self.n_class]),
            tf.reshape(pixel_wise_softmax_2(self._logits), [-1, self.n_class])
            ))

    @property
    def predicter(self):
        return pixel_wise_softmax_2(self._logits)

    @property
    def correct_pred(self):
        return tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))

    @property
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


def create_lstm(xs, n_class, lstm_layers, lstm_filter_size=3, **kwargs):
    input_shape = xs[0].shape.as_list()[1:]
    # out channels = in channels
    channels = input_shape[2]
    filter_shape = [lstm_filter_size, lstm_filter_size, 2*channels, channels]
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    stddev = np.sqrt(2 / (lstm_filter_size**2 * channels))

    def weight_init():
        return weight_variable([lstm_filter_size, lstm_filter_size,
                                2*channels, channels], stddev)

    variables = []
    in_list = xs
    for k_layer in range(lstm_layers):
        fw_cell = crc.BasicConvLSTMCell(input_shape, filter_shape, strides,
                                        padding, weight_init=weight_init)
        bw_cell = crc.BasicConvLSTMCell(input_shape, filter_shape, strides,
                                        padding, weight_init=weight_init)

        with vs.variable_scope('lstm_layer%d' % k_layer) as scope:
            (ys, state_fw, state_bw) = cr.static_bidirectional_rnn(
                fw_cell, bw_cell, in_list, dtype=tf.float32, scope=scope)
            # keep track of variables
            with vs.variable_scope('fw') as fw_scope:
                fw_scope.reuse_variables()
                variables.append(vs.get_variable(crc._BIAS_VARIABLE_NAME))
                variables.append(vs.get_variable(crc._WEIGHTS_VARIABLE_NAME))
            with vs.variable_scope('bw') as bw_scope:
                bw_scope.reuse_variables()
                variables.append(vs.get_variable(crc._BIAS_VARIABLE_NAME))
                variables.append(vs.get_variable(crc._WEIGHTS_VARIABLE_NAME))

        in_list = ys

    # 1x1 convolution to get logits
    out_tensor = tf.concat(in_list, 0)
    weight = weight_variable([1, 1, 2*channels, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(out_tensor, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)

    return output_map, variables


def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16,
                    filter_size=3, pool_size=2, summaries=True, **kwargs):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info("Layers {layers}, features {features}, "
                 "filter size {filter_size}x{filter_size}, "
                 "pool size: {pool_size}x{pool_size}".format(
                    layers=layers, features=features_root,
                    filter_size=filter_size, pool_size=pool_size))

    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]

    # x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
    x_image = x
    in_node = x_image
    batch_size = tf.shape(x_image)[0]

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size,
                                 channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size,
                                 features//2, features], stddev)

        w2 = weight_variable([filter_size, filter_size,
                             features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])

        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        if layer < layers-1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2

    in_node = dw_h_convs[layers-1]

    # up layers
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))

        wd = weight_variable_devonc([pool_size, pool_size,
                                    features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat

        w1 = weight_variable([filter_size, filter_size,
                             features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size,
                             features//2, features//2], stddev)
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])

        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        size *= 2

    # output before mapping to n_class for connection to LSTM
    output_raw = in_node

    # Output Map
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map

    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.summary.image('summary_conv_%02d_01' % i, unet.get_image_summary(c1))
            tf.summary.image('summary_conv_%02d_02' % i, unet.get_image_summary(c2))

        for k in pools.keys():
            tf.summary.image('summary_pool_%02d' % k, unet.get_image_summary(pools[k]))

        for k in deconv.keys():
            tf.summary.image('summary_deconv_concat_%02d' % k, unet.get_image_summary(deconv[k]))

        for k in dw_h_convs.keys():
            tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

        for k in up_h_convs.keys():
            tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_raw, output_map, variables, int(in_size - size)
