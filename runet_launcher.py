from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet_1 import unet
import runet
from tf_unet_1 import util
from data_gen import CTScanTrainDataProvider


if __name__ == '__main__':
    training_iters = 20
    epochs = 20
    dropout = 0.75  # Dropout, probability to keep units
    display_step = 2
    restore = False

    npy_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/'


    generator = CTScanTrainDataProvider(npy_folder, weighting=(0.6, 0.3))
    val_generator = CTScanTrainDataProvider(npy_folder, weighting=(1, 0))
    batch_size = 8

    net = runet.RUnet(batch_size=batch_size,
                      n_lstm_layers=1,
                      channels=generator.channels,
                      n_class=generator.n_class,
                      layers=3,
                      features_root=16,
                      cost="avg_class_ce",
                      cost_kwargs={"class_weights": [1, 10, 25]})

    trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
                           opt_kwargs=dict(momentum=0, learning_rate=0.05))
    path = trainer.train(generator, val_generator, "./runet_trained",
                         training_iters=training_iters,
                         epochs=epochs,
                         dropout=dropout,
                         display_step=display_step,
                         restore=restore)

    x_test, y_test = val_generator(batch_size)
    prediction = net.predict(path, x_test)

    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
