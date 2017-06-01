from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet_1 import unet
from tf_unet_1 import util
from data_gen import CTScanTrainDataProvider
from data_gen import CTScanTestDataProvider


if __name__ == '__main__':
    # training_iters = 20
    # epochs = 20
    # dropout = 0.75  # Dropout, probability to keep units
    # display_step = 2
    # restore = False
    #
    # train_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data_notoken/'
    # test_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 2/npy_data_notoken/'
    # generator = CTScanTrainDataProvider(train_folder, weighting=(0.5, 0.3), use_aug=True)
    # val_generator = CTScanTrainDataProvider(train_folder, weighting=(1, 0))
    # batch_size = 17
    #
    # net = unet.Unet(channels=generator.channels,
    #                 n_class=generator.n_class,
    #                 layers=3,
    #                 features_root=16,
    #                 cost="avg_class_ce",
    #                 cost_kwargs={"class_weights": [1, 5, 10]})
    #
    # trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
    #                        opt_kwargs=dict(momentum=0, learning_rate=0.01))
    #
    # path = trainer.train(generator, val_generator, "./unet_trained",
    #                      training_iters=training_iters,
    #                      epochs=epochs,
    #                      dropout=dropout,
    #                      display_step=display_step,
    #                      restore=restore)
    #
    # test_generator = CTScanTestDataProvider(test_folder)
    # x_test, y_test = test_generator(50)
    # prediction = net.predict(path, x_test)

    training_iters = 20
    epochs1 = 15
    epochs2 = 15
    epochs3 = 15
    dropout = 1  # Dropout, probability to keep units
    display_step = 2

    npy_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 2/npy_data_notoken/'

    generator1 = CTScanTrainDataProvider(npy_folder, weighting=(0.9, 0.05))
    generator2 = CTScanTrainDataProvider(npy_folder, weighting=(0.6, 0.2))
    generator3 = CTScanTrainDataProvider(npy_folder)
    val_generator = CTScanTrainDataProvider(npy_folder, weighting=(1, 0), use_aug=False)
    batch_size = 8

    net = unet.Unet(channels=generator1.channels,
                    n_class=generator1.n_class,
                    layers=3,
                    features_root=16,
                    cost="avg_class_ce",
                    cost_kwargs={"class_weights": [1, 5, 10]})  # bkgd, liver, tumor

    trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
                           opt_kwargs={"momentum": 0,
                                       "learning_rate": 0.15,
                                       "decay_rate": 0.95})

    trainer.train(generator1, val_generator, "./unet_trained/stage1",
                  prediction_path="./unet_prediction/stage1",
                  training_iters=training_iters,
                  epochs=epochs1,
                  dropout=dropout,
                  display_step=display_step,
                  restore=False)

    # net.set_cost("avg_class_ce", {"class_weights": [2, 3, 7]})
    trainer.opt_kwargs["learning_rate"] = 0.05
    # trainer.opt_kwargs["momentum"] = 0

    trainer.train(generator2, val_generator, "./unet_trained/stage2",
                  restore_path="./unet_trained/stage1",
                  prediction_path="./unet_prediction/stage2",
                  training_iters=training_iters,
                  epochs=epochs2,
                  dropout=dropout,
                  display_step=display_step,
                  restore=True)

    # net.set_cost("avg_class_ce", {"class_weights": [1, 1, 9]})
    trainer.opt_kwargs["learning_rate"] = 0.01

    path = trainer.train(generator3, val_generator, "./unet_trained/stage3",
                         restore_path="./unet_trained/stage2",
                         prediction_path="./unet_prediction/stage3",
                         training_iters=training_iters,
                         epochs=epochs3,
                         dropout=dropout,
                         display_step=display_step,
                         restore=True)

    x_test, y_test = val_generator(50)
    prediction = net.predict(path, x_test)
    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
