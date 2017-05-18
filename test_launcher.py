from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet_1 import unet
from tf_unet_1 import util
from data_gen import *

import tensorflow as tf

if __name__ == '__main__':
    training_iters = 20
    epochs = 10
    dropout = 0.75  # Dropout, probability to keep units
    display_step = 2
    restore = False

    generator = CTScanTrainDataProvider('/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/')
    test_generator = CTScanTestDataProvider('/ihome/azhu/cs189/data/liverScans/Training Batch 2/npy_data_notoken/')

    batch_size = 4

    net = unet.Unet(channels=generator.channels,
                    n_class=generator.n_class,
                    layers=3,
                    features_root=16,
                    cost="dice_coefficient")

    # new_graph = tf.Graph()
    # with tf.Session(graph=new_graph) as sess:
    #     saver = tf.train.import_meta_graph('./unet_trained/model.cpkt.meta')
    #     saver.restore(sess, './unet_trained/model.cpkt')

    x_test, y_test = test_generator(20)
    prediction = net.predict('./unet_trained/model.cpkt', x_test)

    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
