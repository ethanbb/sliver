from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet_1 import unet
import runet
from tf_unet_1 import util
from data_gen import CTScanTrainDataProvider

# 2017-05-17 20:29:29,199 Model restored from file: ./runet_trained/model.cpkt
# Testing error rate: 4.99%

if __name__ == '__main__':
    training_iters = 10
    epochs = 10
    dropout = 0.75  # Dropout, probability to keep units
    display_step = 2
    restore = False

    npy_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/'


    generator = CTScanTrainDataProvider(npy_folder)
    batch_size = 4

    net = runet.RUnet(batch_size=batch_size,
                      n_lstm_layers=1,
                      channels=generator.channels,
                      n_class=generator.n_class,
                      layers=3,
                      features_root=16,
                      cost="dice_coefficient")
    net.use_lstm = True

    trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
                           opt_kwargs=dict(momentum=0.2, learning_rate=0.2))
    path = trainer.train(generator, "./runet_trained",
                         training_iters=training_iters,
                         epochs=epochs,
                         dropout=dropout,
                         display_step=display_step,
                         restore=restore)

    x_test, y_test = generator(4)
    prediction = net.predict(path, x_test)

    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
