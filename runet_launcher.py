from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet_1 import unet
import runet
from tf_unet_1 import util
from data_gen import CTScanTrainDataProvider

# 2017-05-17 21:45:20,269 Epoch 9, Average loss: -1.0588, learning rate: 0.1260
# 2017-05-17 21:45:28,383 Verification error= 11.1%, loss= -1.0417
# 2017-05-17 21:45:42,144 Optimization Finished!
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:8a:00.0)
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:1) -> (device: 1, name: Tesla K80, pci bus id: 0000:8b:00.0)
# 2017-05-17 21:45:55,574 Model restored from file: ./runet_trained/model.cpkt
# Testing error rate: 7.53%

if __name__ == '__main__':
    training_iters = 20
    epochs = 20
    dropout = 0.75  # Dropout, probability to keep units
    display_step = 2
    restore = False

    npy_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/'


    generator = CTScanTrainDataProvider(npy_folder, weighting=(0.5, 0.2))
    batch_size = 10

    net = runet.RUnet(batch_size=batch_size,
                      n_lstm_layers=1,
                      channels=generator.channels,
                      n_class=generator.n_class,
                      layers=3,
                      features_root=16,
                      cost="avg_class_ce_symmetric",
                      cost_kwargs={"class_weights": [0, 1, 5]})

    trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
                           opt_kwargs=dict(momentum=0.9, learning_rate=0.05))
    path = trainer.train(generator, "./runet_trained",
                         training_iters=training_iters,
                         epochs=epochs,
                         dropout=dropout,
                         display_step=display_step,
                         restore=restore)

    x_test, y_test = generator(4)
    prediction = net.predict(path, x_test)

    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
