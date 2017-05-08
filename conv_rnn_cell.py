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
        filter_channels_in = input_shape[2] + filter_shape[2]
        filter_shape.insert(2, filter_channels_in)
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._use_cudnn = use_cudnn_on_gpu
        self._activation = activation
        self._scope = scope
        # simulate a convolution to get state and output sizes
        input = array_ops.placeholder(
            dtypes.float32, shape=[None]+input_shape[0:2]+[filter_channels_in])
        filter = array_ops.placeholder(dtypes.float32, shape=filter_shape)
        output = nn_ops.conv2d(input, filter, strides, padding)
        output = array_ops.transpose(output, perm=[0, 3, 1, 2])
        self._output_shape = output.get_shape()[1:]

    @property
    def state_size(self):
        return self._output_shape

    @property
    def output_size(self):
        return self._output_shape

    def __call__(self, input, state, scope=None):
        """Inputs and outputs are of shape [batch, channels, height, width]
        (Note the unusual shape which allows static_bidirectional_rnn to work)
        This is dealt with in the wrappers in core_conv_rnn
        """
        input = array_ops.transpose(input, perm=[0, 2, 3, 1])
        state = array_ops.transpose(state, perm=[0, 2, 3, 1])
        with vs.variable_scope(scope or self._scope):
            output = self._activation(
                _conv2d([input, state], self._filter_shape, self._strides,
                        self._padding, self._use_cudnn, True, scope=scope))
        output = array_ops.transpose(output, perm=[0, 3, 1, 2])
        return output, output


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