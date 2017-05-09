from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np

npy_folder = '/ihome/azhu/cs189/data/liverScans/Training Batch 1/npy_data/'


class CTScanDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavior the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    """
    channels = 1
    n_class = 4

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.volume_index = -1
        self.volume_depth = -1
        self.frame_index = -1
        self.num_samples = 28

    def _load_data_and_label(self):
        data, label = self._next_data()

        nx = data.shape[1]
        ny = data.shape[0]

        train_data = self._process_data(data)
        labels = self._process_labels(label)

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class)

    def _process_labels(self, label):
        nx = label.shape[1]
        ny = label.shape[0]
        labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
        labels[..., 0] = (label == 0)
        labels[..., 1] = (label == 1)
        labels[..., 2] = (label == 2)
        labels[..., 3] = (label == -1)
        return labels

    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        if (np.amax(data) != 0):
            data /= np.amax(data)
        return data

    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation

        :param data: the data array
        :param labels: the label array
        """
        return data, labels

    def __call__(self, n=1):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))

        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels

        return X, Y

    def _cycle_frame(self):
        self.frame_index += 1
        if self.frame_index >= self.volume_depth:
            self.frame_index = 0
            return True
        return False  # returns False if we have not finished the volume yet

    def _next_data(self):
        if (self._cycle_frame()):
            self.data, self.label = self._next_volume()
        return self.data[:, :, self.frame_index], self.label[:, :, self.frame_index]

    def _cycle_volume(self):
        self.volume_index += 1
        if self.volume_index >= self.num_samples:
            self.volume_index = 0

    def _next_volume(self):
        self._cycle_volume()
        data_path = npy_folder + 'volume-' + str(self.volume_index)
        label_path = npy_folder + 'segmentation-' + str(self.volume_index)

        data = np.load(data_path + '.npy')
        label = np.load(label_path + '.npy')

        self.volume_depth = data.shape[2]

        return data, label
