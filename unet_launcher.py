from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet_1 import unet
from tf_unet_1 import util
from data_gen import CTScanTrainDataProvider

# Last time, failed at 2017-05-17 20:40:04,441 Epoch 6, Average loss: -1.1211, learning rate: 0.0735
# 2017-05-17 20:57:09,612 Epoch 19, Average loss: -1.1076, learning rate: 0.0377
# 2017-05-17 20:57:09,968 Verification error= 3.9%, loss= -1.0624
# 2017-05-17 20:57:11,053 Optimization Finished!
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:8a:00.0)
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:1) -> (device: 1, name: Tesla K80, pci bus id: 0000:8b:00.0)
# 2017-05-17 20:57:11,802 Model restored from file: ./unet_trained/model.cpkt
# Testing error rate: 1.64%
# with layers modified

if __name__ == '__main__':
    training_iters = 20
    epochs = 20
    dropout = 0.75  # Dropout, probability to keep units
    display_step = 2
    restore = False

    npy_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/'

    generator = CTScanTrainDataProvider(npy_folder)
    batch_size = 4

    net = unet.Unet(channels=generator.channels,
                    n_class=generator.n_class,
                    layers=3,
                    features_root=16,
                    cost="dice_coefficient")

    trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
                           opt_kwargs=dict(momentum=0.2, learning_rate=0.1))

    path = trainer.train(generator, "./unet_trained",
                         training_iters=training_iters,
                         epochs=epochs,
                         dropout=dropout,
                         display_step=display_step,
                         restore=restore)

    x_test, y_test = generator(4)
    prediction = net.predict(path, x_test)

    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
