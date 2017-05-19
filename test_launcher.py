from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet_1 import unet
from tf_unet_1 import util
from data_gen import *

import tensorflow as tf

if __name__ == '__main__':

    # generator = CTScanTrainDataProvider('/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/')
    test_generator = CTScanTestDataProvider('/ihome/azhu/cs189/data/liverScans/Training Batch 2/npy_data_notoken/')

    batch_size = 4

    net = unet.Unet(channels=test_generator.channels,
                    n_class=test_generator.n_class,
                    layers=3,
                    features_root=16,
                    cost="dice_coefficient")

    test_batch_num = 0
    err = 0.
    x_test, y_test = test_generator(50)
    while (np.any(x_test)):
        test_batch_num += 1
        prediction = net.predict('./unet_trained/model.cpkt', x_test)
        err += unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))
        x_test, y_test = test_generator(50)

    print("Total summed err: {:.2f}".format(err))
    final_err = err / test_batch_num
    print("Testing error rate: {:.2f}%".format(final_err))

# Testing error rate: 6.22% with modified dice coefficient, trained 50 epochs
# Testing error rate: 6.88% with average class pixel accuracy
# Testing error rate: 6.84% with modified dice coefficient unet_trained_newloss
# Testing error rate: 8.80% with unet_trained_1
