from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet import unet
from tf_unet import util
from data_gen import CTScanDataProvider

if __name__ == '__main__':
    nx = 572
    ny = 572

    training_iters = 20
    epochs = 10
    dropout = 0.75  # Dropout, probability to keep units
    display_step = 2
    restore = False

    # generator = image_gen.RgbDataProvider(nx, ny, cnt=20, rectangles=False)
    # need new generator
    generator = CTScanDataProvider()

    net = unet.Unet(channels=generator.channels,
                    n_class=generator.n_class,
                    layers=3,
                    features_root=16,
                    cost="dice_coefficient")

    trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(generator, "./unet_trained",
                         training_iters=training_iters,
                         epochs=epochs,
                         dropout=dropout,
                         display_step=display_step,
                         restore=restore)

    x_test, y_test = generator(4)
    prediction = net.predict(path, x_test)

    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
