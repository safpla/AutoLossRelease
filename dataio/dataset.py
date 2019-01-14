import os, sys
import numpy as np
import json
import random
import tensorflow as tf
from matplotlib import pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)


class Dataset(object):
    def __init__(self):
        pass

    def load_h5py(self, filename):
        f = h5py.File(filename, "r")
        self._dataset_input = f['img']
        self._dataset_target = f['label']
        self._num_examples = len(self._dataset_target)
        print('num of examples: ', self._num_examples)

        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def load_npy(self, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            self._dataset_input = []
            self._dataset_target = []
            for point in data:
                self._dataset_input.append(point['x'])
                self._dataset_target.append(point['y'])
            self._dataset_input = np.array(self._dataset_input)
            self._dataset_target = np.array(self._dataset_target)
            self._num_examples = len(self._dataset_target)
            print('load {} samples.'.format(self._num_examples))

            self._index = np.arange(self._num_examples)
            self._index_in_epoch = 0
            self._epochs_completed = 0

    def load_json(self, filename):
        with open(filename, 'r') as fin:
            data = json.load(fin)
            self._dataset_input = []
            self._dataset_target = []
            for point in data:
                self._dataset_input.append(point['input'])
                self._dataset_target.append(point['target'])
            self._dataset_input = np.array(self._dataset_input)
            self._dataset_target = np.array(self._dataset_target)
            self._num_examples = len(self._dataset_target)
            print('load {} samples.'.format(self._num_examples))
            self._index = np.arange(self._num_examples)
            self._index_in_epoch = 0
            self._epochs_completed = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def build_from_data(self, input, target):
        self._dataset_input = np.array(input)
        self._dataset_target = np.array(target)
        self._num_examples = len(self._dataset_target)
        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def next_batch(self, batch_size, shuffle=True):
        # batch_size is the first dimension
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if shuffle:
                self.shuffle()
            # Start next epoch
            start = 0
            assert batch_size <= self._num_examples
            self._index_in_epoch = start + batch_size
        end = self._index_in_epoch
        batch_index = self._index[start:end]
        #print('start:{}, end:{}'.format(start, end))
        batch_index = list(np.sort(batch_index))
        target = self._dataset_target[batch_index]
        input = self._dataset_input[batch_index]
        samples = {}
        samples['input'] = input
        samples['target'] = target
        return samples

    def reset(self, shuffle=False):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        if shuffle:
            self.shuffle()

    def shuffle(self):
        np.random.shuffle(self._index)

    def resize(self, number, shuffle=False):
        self._dataset_input = self._dataset_input[:number]
        self._dataset_target = self._dataset_target[:number]
        self._num_examples = len(self._dataset_target)
        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0
        if shuffle:
            self.shuffle()


if __name__ == '__main__':
    filename = os.path.join(root_path, 'Data/toy_16_200/train.npy')
    dataset = Dataset()
    dataset.load_npy(filename)
    point = dataset.next_batch(100)
    print('target: ', point['target'])
