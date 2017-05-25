from __future__ import print_function, division, absolute_import, unicode_literals
from elastic_deformation import elastic_transform

import numpy as np
import random
import warnings

class CTScanTestDataProvider(object):
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
    n_class = 3

    def __init__(self, npy_folder, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.volume_index = 27
        self.volume_depth = -1
        self.frame_index = -2
        self.num_samples = 130
        self.npy_folder = npy_folder
        self.data = None
        self.label = None

    def _load_data_and_label(self):
        data, label = self._next_data()
        if (not np.any(data)):
            return False, False
        # print ("volume index: " + str(self.volume_index) + ", frame index: " + str(self.frame_index))
        # max_label = np.amax(label)
        # print ("max label of this frame: " + str(max_label))

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
        # labels[..., 3] = (label == -1)
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

    def __call__(self, n=4):
        train_data, labels = self._load_data_and_label()
        if (not np.any(train_data)):
            return False, False
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))

        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            if (not np.any(train_data)):
                return False, False
            X[i] = train_data
            Y[i] = labels
        # print(self.volume_index, self.frame_index)
        return X, Y

    def _cycle_frame(self):
        self.frame_index += 1
        if (self.frame_index >= self.volume_depth or self.frame_index == -1):
            self.frame_index = 0
            return True
        return False  # returns False if volume is unfinished

    def _next_data(self):
        if (self._cycle_frame()):
            self._next_volume()
            if (not np.any(self.data)):
                return False, False  # returns False if entire batch is finished
        return self.data[:, :, self.frame_index], self.label[:, :, self.frame_index]

    def _cycle_volume(self):
        self.volume_index += 1
        if self.volume_index >= self.num_samples:
            return True  # returns True if entire batch is finished

    def _next_volume(self):
        if (self._cycle_volume()):
            self.data, self.label = False, False
            return
        data_path = self.npy_folder + 'volume-' + str(self.volume_index)
        label_path = self.npy_folder + 'segmentation-' + str(self.volume_index)

        data = np.load(data_path + '.npy')
        label = np.load(label_path + '.npy')

        self.volume_depth = data.shape[2]
        self.data, self.label = data, label


class CTScanTrainDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavior the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param npy_folder: the folder containing liver scans and labels in npy format
    :param weighting: (optional) if specified, a pair consisting of:
        - probability of returning a subvolume containing tumor
        - probability of returning a subvolume containing liver but no tumor
        Returns a subvolume contianing only background with the remaining probability.
        If not specified, picks subvolumes at random.
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    """
    channels = 1
    n_class = 3

    def __init__(self, npy_folder, weighting=None, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.volume_index = -1
        # boolean arrays of what each frame contains
        self.unused_frames = []
        self.bg_frames = []
        self.liver_frames = []  # contain liver but no tumor
        self.tumor_frames = []
        self.num_samples = 28
        self.npy_folder = npy_folder
        self.weighting = weighting
        self.no_weighting = weighting is None

    def _load_data_and_label(self):
        data, label = self._next_data()

        # print ("volume index: " + str(self.volume_index) + ", frame index: " + str(self.frame_index))
        # max_label = np.amax(label)
        # print ("max label of this frame: " + str(max_label))

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
        # labels[..., 3] = (label == -1)
        return labels

    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        if (np.amax(data) != 0):
            data /= np.amax(data)
        return data

    # def _post_process(self, data, labels):
    #     """
    #     Post processing hook that can be used for data augmentation
    #
    #     :param data: the data array
    #     :param labels: the label array
    #     """
    #     return data, labels
    def _augment_data(self, data, labels):
        """
        Post processing hook to perform elastic transforms

        :param data: the data array
        :param labels: the label array
        """
        aug_data, aug_labels = elastic_transform(data, labels,
                                                 data.shape[1] * 2,
                                                 data.shape[1] * 0.08)
        return aug_data, aug_labels

    def __call__(self, n):
        if self.volume_index == -1:
            self.data, self.label = self._next_volume()

        num_frames = len(self.unused_frames)
        if n > num_frames:
            warnings.warn('Batch size is larger than volume; padding with zeros')
            X, Y = self(num_frames)
            nx = X.shape[1]
            ny = Y.shape[2]

            X = np.concatenate((X, np.zeros((n - num_frames, nx, ny, self.channels))))
            Y = np.concatenate((Y, np.zeros((n - num_frames, nx, ny, self.n_class))))
            Y[num_frames:, :, :, 0] = 1

            return X, Y

        window = np.full(n, 1)
        valid_start = np.convolve(self.unused_frames, window, 'valid') == n

        if not self.no_weighting:
            skew = random.random()
            if skew < self.weighting[0]:
                # with tumor
                has_tumor = np.convolve(self.tumor_frames, window, 'valid') > 0
                valid_start &= has_tumor
            elif skew < (self.weighting[0] + self.weighting[1]):
                # liver, no tumor
                has_tumor = np.convolve(self.tumor_frames, window, 'valid') > 0
                has_liver = np.convolve(self.liver_frames, window, 'valid') > 0
                valid_start = valid_start & ~has_tumor & has_liver
            else:
                # no liver or tumor
                all_bg = np.convolve(self.bg_frames, window, 'valid') == n
                valid_start &= all_bg

        valid_start = np.nonzero(valid_start)[0]
        if len(valid_start) == 0:
            self.data, self.label = self._next_volume()
            return self(n)

        start = np.random.choice(valid_start)
        self.frame_index = start - 1
        slice_range = range(start, start + n)
        self.unused_frames[slice_range] = False

        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        # X = np.zeros((n, nx, ny, self.channels))
        # Y = np.zeros((n, nx, ny, self.n_class))
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))

        X[0] = train_data
        Y[0] = labels

        # aug_data, aug_labels = self._augment_data(train_data, labels)
        # X[1] = aug_data
        # Y[1] = aug_labels
        m = int(n / 2)
        for i in range(1, m):
            train_data, labels = self._load_data_and_label()
            # X[i] = train_data
            # Y[i] = labels

            aug_data, aug_labels = self._augment_data(train_data, labels)
            X[2*i] = train_data
            Y[2*i] = labels
            X[2*i+1] = aug_data
            Y[2*i+1] = aug_labels

        return X, Y

    def _next_data(self):
        self.frame_index += 1
        return self.data[:, :, self.frame_index], self.label[:, :, self.frame_index]

    def _cycle_volume(self):
        self.volume_index += 1
        if self.volume_index >= self.num_samples:
            self.volume_index = 0

    def _next_volume(self):
        self._cycle_volume()
        data_path = self.npy_folder + 'volume-' + str(self.volume_index)
        label_path = self.npy_folder + 'segmentation-' + str(self.volume_index)

        data = np.load(data_path + '.npy')
        label = np.load(label_path + '.npy')

        self.unused_frames = np.full(np.shape(label)[2], True)

        if not self.no_weighting:
            self.liver_frames = np.full(np.shape(label)[2], False)
            self.tumor_frames = np.full(np.shape(label)[2], False)

            self.tumor_frames[np.unique(np.where(label == 2)[2])] = True
            self.liver_frames[np.unique(np.where(label == 1)[2])] = True
            self.liver_frames &= ~self.tumor_frames
            self.bg_frames = ~self.tumor_frames & ~self.liver_frames

        return data, label
