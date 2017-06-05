from __future__ import print_function, division, absolute_import, unicode_literals
from runet.tf_unet_1 import unet
from runet import runet
from runet.tf_unet_1 import util
from runet.data_gen import CTScanTrainDataProvider


# Modify function call at bottom of file for transfer learning.
# note: freeze is ignored unless transfer is true.
def train_runet(transfer=False, freeze=False):
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

    weights = [1, 3, 5] if transfer else [1, 1, 1]

    net = runet.RUnet(batch_size=batch_size,
                      n_lstm_layers=2,
                      channel_mult=[1.5, 2],
                      channels=generator1.channels,
                      n_class=generator1.n_class,
                      layers=3,
                      features_root=16,
                      cost="avg_class_ce",
                      cost_kwargs={"class_weights": weights})  # bkgd, liver, tumor

    trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
                           opt_kwargs={"momentum": 0,
                                       "learning_rate": 0.15,
                                       "decay_rate": 0.95})

    if transfer:
        net.restore_saver = net.transfer_saver

    var_list = net.nontransfer_variables if transfer and freeze else None

    trainer.train(generator1, val_generator, "./runet_trained/stage1",
                  restore_path="./transfer",
                  prediction_path="./runet_prediction/stage1",
                  training_iters=training_iters,
                  epochs=epochs1,
                  dropout=dropout,
                  display_step=display_step,
                  restore=transfer,
                  var_list=var_list)

    trainer.opt_kwargs["learning_rate"] = 0.05

    net.restore_saver = net.saver

    trainer.train(generator2, val_generator, "./runet_trained/stage2",
                  restore_path="./runet_trained/stage1",
                  prediction_path="./runet_prediction/stage2",
                  training_iters=training_iters,
                  epochs=epochs2,
                  dropout=dropout,
                  display_step=display_step,
                  restore=True,
                  var_list=var_list)

    trainer.opt_kwargs["learning_rate"] = 0.01

    path = trainer.train(generator3, val_generator, "./runet_trained/stage3",
                         restore_path="./runet_trained/stage2",
                         prediction_path="./runet_prediction/stage3",
                         training_iters=training_iters,
                         epochs=epochs3,
                         dropout=dropout,
                         display_step=display_step,
                         restore=True,
                         var_list=var_list)


if __name__ == '__main__':
    train_runet(transfer=False, freeze=False)
