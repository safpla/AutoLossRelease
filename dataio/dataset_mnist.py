import os, sys
import numpy as np
import random
import tensorflow as tf

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from dataio.dataset import Dataset
from tensorflow.examples.tutorials.mnist import input_data

class Dataset_mnist(Dataset):
    def __init__(self):
        pass

    def load_mnist(self, data_dir, mode=tf.estimator.ModeKeys.TRAIN):
        mnist = input_data.read_data_sets(data_dir, one_hot=True)
        if mode == tf.estimator.ModeKeys.TRAIN:
            _dataset = mnist.train
        else:
            _dataset = mnist.test
        self._dataset_input = _dataset.images
        self._dataset_target = _dataset.labels
        self._num_examples = len(self._dataset_target)
        print('Load {} samples.'.format(self._num_examples))
        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0


if __name__ == '__main__':
    dataset = Dataset_mnist()
    dataset.load_mnist(mode=tf.estimator.ModeKeys.TRAIN)
