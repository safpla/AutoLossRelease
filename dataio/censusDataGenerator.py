""" Download and clean census dataset."""

# __author__ == 'Haowen Xu'
# __data__ == '04_25_2018'

import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
import numpy as np
from six.moves import urllib
import json
import utils
import random

config_path = os.path.join(root_path, 'config/classification.cfg')
config = utils.Parser(config_path)

data_train = config.train_data_file
data_valid = config.valid_data_file
data_test = config.test_data_file

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

def _download_and_clean_file(mode):
    if mode == 'train_valid':
        url = TRAINING_URL
    else:
        url = EVAL_URL
    temp_file, _ = urllib.request.urlretrieve(url)
    samples = []
    with open(temp_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace(', ', ',')
            if not line or ',' not in line:
                continue
            if line[-1] == '.':
                line = line[:-1]
                print('end with dot')
            s = line.split(',')
            sample = {'x': s[:-1], 'y': s[-1]}
            samples.append(sample)
    if mode == 'train_valid':
        random.shuffle(samples)
        total_num = len(samples)
        train_num = int(total_num * 0.9)
        print(total_num)
        trainset = samples[:train_num]
        validset = samples[train_num:]
        with open(data_train, 'wb') as f:
            np.save(f, trainset)
        with open(data_valid, 'wb') as f:
            np.save(f, validset)
    else:
        random.shuffle(samples)
        print(len(samples))
        with open(data_test, 'wb') as f:
            np.save(f, samples)


def main():
    train_data_file = config.train_data_file
    valid_data_file = config.valid_data_file
    test_data_file = config.test_data_file
    data_folder = '/'.join(train_data_file.split('/')[:-1])
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    #_download_and_clean_file('train_valid')
    _download_and_clean_file('test')


if __name__ == '__main__':
    main()
