# NOTE: TensorFlow version must be 1.0.1
import tensorflow as tf
import conv_rnn_cell
import conv_rnn

sess = tf.InteractiveSession()

time_points = 20

# inputs
input_shape = [14, 15, 3]

x = [tf.placeholder(tf.float32, shape=[None] + input_shape)
     for t in range(time_points)]

filters_fw = 10
filters_bw = 20

cell_fw = conv_rnn_cell.BasicConvRNNCell(
    input_shape, [3, 3, filters_fw], [1, 1, 1, 1], 'SAME')
cell_bw = conv_rnn_cell.BasicConvRNNCell(
    input_shape, [3, 3, filters_bw], [1, 1, 1, 1], 'SAME')

(y, state_fw, state_bw) = conv_rnn.static_bidirectional_rnn(
    cell_fw, cell_bw, x, dtype=tf.float32)

# y, state_fw, and state_bw should be the correct shapes!
