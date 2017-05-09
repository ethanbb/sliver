"""Module implementing convolutional RNN and LSTM cells
Targeted at TensorFlow version 1.0.1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from tensorflow.python.util import nest

from tensorflow.contrib.rnn import LSTMStateTuple

_BIAS_VARIABLE_NAME = "biases"
_FILTERS_VARIABLE_NAME = "filters"


class BasicConvRNNCell(RNNCell):
    """Basic RNN cell using 2D convolution
    Based on BasicRNNCell

    Args:
        input_shape: [in_height, in_width, in_channels]
        filter_shape: [filter_height, filter_width, out_channels]
            (input channels of filter is automatically calculated)
        strides: (see 'strides' input to tf.nn.conv2d)
        padding: padding algorithm, either 'SAME' or 'VALID'
        use_cudnn_on_gpu: (default: True) see tf.nn.conv2d
        activation: (default: tanh) activation function to use
        scope: (optional) the scope to use for parameters
    """

    def __init__(self, input_shape, filter_shape, strides, padding,
                 use_cudnn_on_gpu=True, activation=tanh,
                 scope='basic_conv_rnn_cell'):
        self._input_shape = input_shape
        self._filter_channels_in = input_shape[2] + filter_shape[2]
        filter_shape.insert(2, self._filter_channels_in)
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._use_cudnn = use_cudnn_on_gpu
        self._activation = activation
        self._scope = scope
        dummy_output = self._dummy_forward()
        self._output_shape = dummy_output.get_shape()[1:]

    def _dummy_forward(self):
        """Do a forward pass on placeholders to determine output size"""
        combined_in = array_ops.placeholder(
            dtypes.float32,
            shape=[None] + self._input_shape[0:2] + [self._filter_channels_in])
        filt = array_ops.placeholder(dtypes.float32, shape=self._filter_shape)
        output = nn_ops.conv2d(combined_in, filt, self._strides, self._padding)
        return transpose(output, [0, 3, 1, 2])

    @property
    def state_size(self):
        return self._output_shape

    @property
    def output_size(self):
        return self._output_shape

    def __call__(self, x, state, scope=None):
        """Inputs and outputs are of shape [batch, channels, height, width]
        (Note the unusual shape which allows static_bidirectional_rnn to work)
        This is dealt with in the wrappers in core_conv_rnn
        """
        scope = scope or self._scope
        with vs.variable_scope(scope):
            x = transpose(x, [0, 2, 3, 1])
            state = transpose(state, [0, 2, 3, 1])
            output = self._activation(
                _conv2d([x, state], self._filter_shape, self._strides,
                        self._padding, self._use_cudnn, True, scope=scope))
        output = transpose(output, [0, 3, 1, 2])
        return output, output


class BasicConvLSTMCell(BasicConvRNNCell):
    """LSTM version of BasicConvRNNCell
    Args that differ from BasicConvRNNCell:
        forget_bias: (default: 1.0) special initial bias for forget gates
        activation: this is the activation for j and new_c (gates use sigmoid)
        filter_shape: last element is the number of filters in output (not *4)
    """
    def __init__(self, input_shape, filter_shape, strides, padding,
                 forget_bias=1.0, use_cudnn_on_gpu=True, activation=tanh,
                 scope='basic_conv_lstm_cell'):
        super(BasicConvLSTMCell, self).__init__(
            input_shape, filter_shape, strides, padding, use_cudnn_on_gpu,
            activation, scope)
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return LSTMStateTuple(self._output_shape, self._output_shape)

    def __call__(self, x, state, scope=None):
        scope = scope or self._scope
        with vs.variable_scope(scope):
            c, h = state
            # transpose to [batch, height, width, channel]
            x = transpose(x, [0, 2, 3, 1])
            c, h = transpose(state, [0, 2, 3, 1])

            filter_shape = self._filter_shape[0:3] + [4*self._filter_shape[3]]
            concat = _conv2d([x, h], filter_shape, self._strides,
                             self._padding, self._use_cudnn, True, scope=scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4,
                                         axis=3)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            # transpose to [batch, channel, height, width]
            new_c = transpose(new_c, [0, 3, 1, 2])
            new_h = transpose(new_h, [0, 3, 1, 2])

            new_state = LSTMStateTuple(new_c, new_h)
            return new_h, new_state


def _conv2d(args, filter_shape, strides, padding, use_cudnn_on_gpu, bias,
            bias_start=0.0, scope=None):
    """2D convolution with newly-created or stored filters

    Args:
        args: a 4D Tensor or list of Tensors, each [batch x h x w x channels]
        filter_shape: dimensions of filter ([h, w, in_channels, out_channels])
        strides, padding, use_cudnn_on_gpu: see BasicConvRNNCell
        bias: boolean, whether to add a bias term or not
        bias_start: starting value to initialize the bias; 0 by default
        scope: (optional) Variable scope to create parameters in

    Returns:
        A 4D Tensor which is the result of applying conv2d to the inputs
        (concatenated along their channel dimension) with the filter.

    Raises:
        ValueError: if some of the arguments have unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    dtype = [a.dtype for a in args][0]

    # do computation
    if scope is None:
        scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        filters = vs.get_variable(
            _FILTERS_VARIABLE_NAME, filter_shape, dtype=dtype)
        if len(args) == 1:
            res = nn_ops.conv2d(args[0], filters, strides,
                                padding, use_cudnn_on_gpu)
        else:
            res = nn_ops.conv2d(array_ops.concat(args, 3), filters, strides,
                                padding, use_cudnn_on_gpu)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            num_filters = filter_shape[-1]
            biases = vs.get_variable(
                _BIAS_VARIABLE_NAME, [num_filters], dtype=dtype,
                initializer=init_ops.constant_initializer(
                    bias_start, dtype=dtype))
        return nn_ops.bias_add(res, biases)


def transpose(state, perm):
    if isinstance(state, LSTMStateTuple):
        c, h = state
        return LSTMStateTuple(transpose(c, perm), transpose(h, perm))
    else:
        return array_ops.transpose(state, perm=perm)
