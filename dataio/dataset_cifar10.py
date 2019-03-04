import os
import sys
import random
import pickle
import urllib.request
import shutil
import tarfile

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from dataio.dataset import Dataset


def unpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data'], data['labels']

def _extract(tar_path, target_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        file_names = tar.getnames()
        print(file_names)
        for file_name in file_names:
            tar.extract(file_name, target_path)

class Dataset_cifar10(Dataset):
    def __init__(self):
        pass

    def download_and_extract_cifar10(self, data_dir):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        with urllib.request.urlopen(url) as response:
            with open(data_dir, 'wb') as f:
                shutil.copyfileobj(response, f)
        file_name = data_dir.split('/')
        assert file_name == 'cifar-10-batches-py'
        father_dir = '/'.join(data_dir.split('/')[:-1])
        _extract(data_dir, father_dir)

    def load_cifar10(self, data_dir, mode=tf.estimator.ModeKeys.TRAIN):
        if mode == tf.estimator.ModeKeys.TRAIN:
            filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                         'data_batch_4', 'data_batch_5']
        else:
            filenames = ['test_batch']
        self._dataset_input = []
        self._dataset_target = []
        for filename in filenames:
            input, target = unpickle(os.path.join(data_dir, filename))
            self._dataset_input.append(input)
            self._dataset_target.append(target)
        self._dataset_input = np.concatenate(self._dataset_input, axis=0)
        self._dataset_target = np.concatenate(self._dataset_target, axis=0)
        self._num_examples = len(self._dataset_target)
        print('Load {} samples.'.format(self._num_examples))
        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0
