from __future__ import print_function, division, absolute_import, unicode_literals
from data_gen import CTScanTestDataProvider
import tensorflow as tf
import numpy as np
import runet
from tf_unet_1 import unet

def get_performance(net, model_path):
    generator = CTScanTestDataProvider(npy_folder)
    (data, gt) = generator(batch_size)  # load first volume
    accuracies = []
    liver_dices = []
    tumor_dices = []

    # restore model
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        net.restore(sess, model_path)

        def predict(x, y):
            return sess.run(net.predicter,
                            feed_dict={net.x: x, net.y: y, net.keep_prob: 1.})

        while True:
            (data, gt) = generator(batch_size)
            if data is False:
                break
            predictions = []
            gts = []
            while data is not False:
                print(generator.volume_index)
                predictions.append(predict(data, gt))
                gts.append(gt)
                (data, gt) = generator(batch_size)

            prediction = np.concatenate(predictions, axis=0)
            gt = np.concatenate(gts, axis=0)

            accuracies.append(unet.error_rate(prediction, gt))

            eps = 1e-5
            prediction_dense = np.argmax(prediction, axis=3)
            gt_dense = np.argmax(gt, axis=3)

            # liver dice
            prediction_b = prediction_dense > 0
            gt_b = gt_dense > 0

            intersection = np.count_nonzero(prediction_b & gt_b)
            size_pred = np.count_nonzero(prediction_b)
            size_gt = np.count_nonzero(gt_b)

            liver_dices.append(2. * intersection / (size_pred + size_gt + eps))

            # tumor dice
            prediction_b = prediction_dense > 1
            gt_b = gt_dense > 1

            intersection = np.count_nonzero(prediction_b & gt_b)
            size_pred = np.count_nonzero(prediction_b)
            size_gt = np.count_nonzero(gt_b)

            tumor_dices.append(2. * intersection / (size_pred + size_gt + eps))

    mean_acc = np.mean(accuracies)
    mean_ld = np.mean(liver_dices)
    mean_td = np.mean(tumor_dices)

    return (mean_acc, mean_ld, mean_td)

# easy changes here to test the unet
if __name__ == '__main__':
    npy_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/'

    batch_size = 8

    net = runet.RUnet(batch_size=batch_size,
                      n_lstm_layers=2,
                      channel_mult=[1.5, 2],
                      channels=1,
                      n_class=3,
                      layers=3,
                      features_root=16,
                      cost="avg_class_ce",
                      cost_kwargs={})

    # ckpt = tf.train.get_checkpoint_state("./runet_trained/stage1")
    # acc1, ld1, td1 = get_performance(net, ckpt.model_checkpoint_path)
    #
    # ckpt = tf.train.get_checkpoint_state("./runet_trained/stage2")
    # acc2, ld2, td2 = get_performance(net, ckpt.model_checkpoint_path)

    ckpt = tf.train.get_checkpoint_state("./runet_trained/stage3")
    acc3, ld3, td3 = get_performance(net, ckpt.model_checkpoint_path)

    # print("First stage performance:")
    # print("Acc:", acc1, "Liver Dice:", ld1, "Tumor Dice:", td1)
    # print("Second stage performance:")
    # print("Acc: ", acc2, "Liver Dice: ", ld2, "Tumor Dice: ", td2)
    print("Third stage performance:")
    print("Acc:", acc3, "Liver Dice:", ld3, "Tumor Dice:", td3)
