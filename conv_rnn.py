"""Wraps TensorFlow's static_rnn and static_bidirectional_rnn in core_rnn.py
to make them convolution-friendly
"""

from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn
import conv_rnn_cell as crc


def static_rnn(cell, inputs, initial_state=None, dtype=None,
               sequence_length=None, scope=None):
    """Assumption here: if cell is convolutional,
    inputs is a list of [batch x height x width x channels] tensors
    """
    if isinstance(cell, crc.BasicConvRNNCell):
        inputs = [array_ops.transpose(input, perm=[0, 3, 1, 2])
                  for input in inputs]
        if initial_state is not None:
            initial_state = array_ops.transpose(initial_state,
                                                perm=[0, 3, 1, 2])

        (outputs, state) = rnn.static_rnn(
            cell, inputs, initial_state, dtype, sequence_length, scope)

        outputs = [array_ops.transpose(output, perm=[0, 2, 3, 1])
                   for output in outputs]
        state = array_ops.transpose(state, perm=[0, 2, 3, 1])
        return (outputs, state)


def static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                             initial_state_fw=None, initial_state_bw=None,
                             dtype=None, sequence_length=None, scope=None):
    """Assumption here: if cell_fw is convolutional, so is cell_bw and
    inputs is a list of [batch x height x width x channels] tensors
    """
    if isinstance(cell_fw, crc.BasicConvRNNCell):
        inputs = [array_ops.transpose(input, perm=[0, 3, 1, 2])
                  for input in inputs]
        if initial_state_fw is not None:
            initial_state_fw = array_ops.transpose(initial_state_fw,
                                                   perm=[0, 3, 1, 2])
        if initial_state_bw is not None:
            initial_state_bw = array_ops.transpose(initial_state_bw,
                                                   perm=[0, 3, 1, 2])

        (outputs, state_fw, state_bw) = rnn.static_bidirectional_rnn(
            cell_fw, cell_bw, inputs, initial_state_fw, initial_state_bw,
            dtype, sequence_length, scope)

        outputs = [array_ops.transpose(output, perm=[0, 2, 3, 1])
                   for output in outputs]
        state_fw = array_ops.transpose(state_fw, perm=[0, 2, 3, 1])
        state_bw = array_ops.transpose(state_bw, perm=[0, 2, 3, 1])
        return (outputs, state_fw, state_bw)
