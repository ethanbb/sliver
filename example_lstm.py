import tensorflow as tf

sess = tf.InteractiveSession()

time_points = 20;

# inputs
x = [tf.placeholder(tf.float32, shape=[None, 100]) for t in range(time_points)]

fw_units = 20
bw_units = 20

cell_fw = tf.contrib.rnn.BasicLSTMCell(fw_units)
cell_bw = tf.contrib.rnn.BasicLSTMCell(bw_units)
y, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, x, dtype=tf.float32)

# y is a list of length time_points of tensors with fw_units + bw_units features each.
