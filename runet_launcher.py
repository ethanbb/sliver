from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet_1 import unet
import runet
from tf_unet_1 import util
from data_gen import CTScanTrainDataProvider


if __name__ == '__main__':
    training_iters = 20
    epochs1 = 15
    epochs2 = 15
    epochs3 = 15
    dropout = 0.75  # Dropout, probability to keep units
    display_step = 2

    npy_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/'

    generator1 = CTScanTrainDataProvider(npy_folder, weighting=(1, 0))
    generator2 = CTScanTrainDataProvider(npy_folder, weighting=(0.5, 0.3))
    generator3 = CTScanTrainDataProvider(npy_folder)
    val_generator = CTScanTrainDataProvider(npy_folder, weighting=(1, 0))
    batch_size = 10

    net = runet.RUnet(batch_size=batch_size,
                      n_lstm_layers=1,
                      channels=generator1.channels,
                      n_class=generator1.n_class,
                      layers=3,
                      features_root=16,
                      cost="avg_class_ce",
                      cost_kwargs={"class_weights": [1, 6, 9]})

    trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum", prediction_path="prediction/stage1",
                            opt_kwargs=dict(momentum=0, learning_rate=0.05))
    trainer.train(generator1, val_generator, "./runet_trained",
                   training_iters=training_iters,
                   epochs=epochs1,
                   dropout=dropout,
                   display_step=display_step,
                   restore=False)

    net.set_cost("avg_class_ce", {"class_weights": [1, 10, 15]})

    trainer.train(generator2, val_generator, "./runet_trained",
                  training_iters=training_iters,
                  epochs=epochs2,
                  dropout=dropout,
                  display_step=display_step,
                  restore=True)

    net.set_cost("avg_class_ce", {"class_weights": [1, 15, 25]})

    path = trainer.train(generator3, val_generator, "./runet_trained",
                          training_iters=training_iters,
                          epochs=epochs3,
                          dropout=dropout,
                          display_step=display_step,
                          restore=True)

    x_test, y_test = val_generator(batch_size)
    prediction = net.predict(path, x_test)

    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
