from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import random
import os
import math

LIVER_DIR = '/idata/cs189/data/ct/liver_npy/'
STOMACH_DIR = '/idata/cs189/data/ct/stomach_npy/'
DATA_SHAPE = (512, 512)


class LiverStomachDataProvider(object):
    """
    Generates images and labels randomly selected from the set of liver tumor
    and stomach tumor CT scans.
    """
    channels = 1
    n_class = 2

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.liver_index = -1
        self.stomach_index = -1
        self.num_liver_samples = len(os.listdir(LIVER_DIR))
        self.num_stomach_samples = len(os.listdir(STOMACH_DIR))
        random.seed()
        self.liver_order = range(self.num_liver_samples)
        random.shuffle(self.liver_order)
        self.stomach_order = range(self.num_stomach_samples)
        random.shuffle(self.stomach_order)

    def _next_data(self, is_liver):
        if is_liver:
            self.liver_index += 1
            if self.liver_index >= self.num_liver_samples:
                self.liver_index = 0
                random.shuffle(self.liver_order)
            source_dir = LIVER_DIR
            source_num = self.liver_order[self.liver_index]
        else:
            self.stomach_index += 1
            if self.stomach_index >= self.num_stomach_samples:
                self.stomach_index = 0
                random.shuffle(self.stomach_order)
            source_dir = STOMACH_DIR
            source_num = self.stomach_order[self.stomach_index]

        source_file = source_dir + '%d.npy' % source_num
        return np.load(source_file)

    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        if (np.amax(data) != 0):
            data /= np.amax(data)
        return data

    def _load_data(self, is_liver):
        data = self._next_data(is_liver)
        data = self._process_data(data)
        return data.reshape((1,) + DATA_SHAPE + (self.channels,))

    def __call__(self, n):
        X = np.zeros((n,) + DATA_SHAPE + (self.channels,))
        Y = np.zeros((n, self.n_class))

        for i in range(n):
            is_liver = random.random() > 0.5
            if is_liver:
                Y[i, 0] = 1
            else:
                Y[i, 1] = 1

            data_slice = self._load_data(is_liver)

            # agument by rotating and flipping
            nrot = random.randint(0, 3)
            data_slice = np.rot90(data_slice, nrot, (1, 2))

            for axis in [1, 2]:
                if random.random() > 0.5:
                    data_slice = np.flip(data_slice, axis)

            X[i] = data_slice

        return X, Y
