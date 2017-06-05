# sliver
Deep network for segmenting liver lesions from CT scans

## End-to-end training

### U-Net

- Train model: `python unet_launcher.py`
- Test model: `python unet_test.py`

### RU-Net

- Train model: `python runet_launcher.py`
- Test model: `python runet_test.py`

## Separate training of U-Net and RU-Net

- If freezing U-Net weights, set `freeze_unet` in the last line of
`pretrain_runet_launcher.py` to `True`; else ensure it is `False`.
- Train model: `python pretrain_runet_launcher.py`
- Test model: `python runet_test.py`

## Transfer learning

- Train classifier on liver and stomach data: `python transfer_launcher.py`
- In `runet_launcher.py`, modify call at bottom of file to set `transfer=True`,
as well as `freeze=True` if freezing transferred weights.
- Train model: `python runet_launcher.py`
- Test model: `python runet_test.py`

## Acknowledgements

The U-Net base implementation (in `runet/tf_unet_1` and replicated in
`runet/runet.py` and `runet/pretrain_liv_stom/sparse_unet.py` is a modified
version of Joel Akeret's TensorFlow U-Net implementation
[tf_unet](https://github.com/jakeret/tf_unet).

The convolutional LSTM defined in `runet/conv_rnn.py` and
`runet/conv_rnn_cell.py` is based on the LSTM implementations in Google's
TensorFlow codebase.

The elastic deformation code in `runet/elastic_deformation.py` is a Modified
version of [https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a](https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a),
based on Simard, Steinkraus and Platt, "Best Practices for
     Convolutional Neural Networks applied to Visual Document Analysis", in
     Proc. of the International Conference on Document Analysis and
     Recognition, 2003.

Other contributions are cited in their respective files.
