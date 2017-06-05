from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from runet.pretrain_liv_stom import sparse_unet
from runet.pretrain_liv_stom.gen_data import LiverStomachDataProvider
from runet import runet
from runet.data_gen import CTScanTrainDataProvider

if __name__ == '__main__':
    batch_size = 25
    num_samples = 17000  # (roughly, doubling size of the smaller dataset)
    batches_per_run = num_samples // batch_size
    runs = 10
    batches_per_epoch = 100
    epochs = runs * batches_per_run // batches_per_epoch
    display_step = 5
    dropout = 1.  # Dropout, probability to keep units
    weight_decay = 1e-4

    generator = LiverStomachDataProvider()

    net = sparse_unet.SparseUnet(channels=generator.channels,
                                 n_class=generator.n_class,
                                 weight_decay=weight_decay,
                                 features_root=16)

    trainer = sparse_unet.Trainer(net, batch_size=batch_size,
                                  optimizer="momentum",
                                  opt_kwargs={"momentum": 0,
                                              "learning_rate": 0.01,
                                              "decay_rate": 0.95})

    trainer.train(generator, "./sparse_trained", transfer_path="./transfer",
                  training_iters=batches_per_epoch, epochs=epochs,
                  dropout=dropout, display_step=display_step)
