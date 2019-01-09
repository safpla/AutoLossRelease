""" Generate toy dataset for linear classification with L1, L2 regularization """

# __author__ == 'Haowen Xu'
# __data__ == '04_25_2018'

import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
import numpy as np
import json
import utils
import sklearn.datasets as d

config_path = os.path.join(root_path, 'config/classification.cfg')
config = utils.Parser(config_path)
dim = config.dim_input_stud
num_train = config.num_sample_train
num_valid = config.num_sample_valid
num_test = config.num_sample_test
num_train_stud = config.num_sample_train_stud
data_train = config.train_data_file
data_valid = config.valid_data_file
data_test = config.test_data_file
data_train_stud = config.train_stud_data_file
np.random.seed(config.random_seed)


def linear_func(x, w):
    return np.dot(x, w)

def old_main():
    n_samples = num_train + num_valid + num_test + num_train_stud
    cls_set = d.make_classification(n_samples=n_samples,
                                    n_features=dim,
                                    n_informative=2,
                                    n_clusters_per_class=2,
                                    n_redundant=2)
    features = cls_set[0]
    labels = cls_set[1]
    folder = '/'.join(data_train.split('/')[:-1])
    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(data_train, 'wb') as f:
        data = []
        features = cls_set[0][:num_train]
        labels = cls_set[1][:num_train]
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

    with open(data_valid, 'wb') as f:
        data = []
        features = cls_set[0][num_train : num_train + num_valid]
        labels = cls_set[1][num_train : num_train + num_valid]
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

    with open(data_test, 'wb') as f:
        data = []
        features = cls_set[0][num_train+num_valid : num_train+num_valid+num_test]
        labels = cls_set[1][num_train+num_valid : num_train+num_valid+num_test]
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

    with open(data_train_stud, 'wb') as f:
        data = []
        features = cls_set[0][num_train+num_valid+num_test :]
        labels = cls_set[1][num_train+num_valid+num_test :]
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

def main():
    n_samples = num_train + num_valid + num_test
    cls_set = d.make_classification(n_samples=n_samples,
                                    n_features=dim,
                                    n_informative=2,
                                    n_clusters_per_class=2,
                                    n_redundant=2)
    features = cls_set[0]
    labels = cls_set[1]
    folder = '/'.join(data_train.split('/')[:-1])
    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(data_train, 'wb') as f:
        data = []
        features = cls_set[0][:num_train]
        labels = cls_set[1][:num_train]
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

    with open(data_valid, 'wb') as f:
        data = []
        features = cls_set[0][num_train : num_train + num_valid]
        labels = cls_set[1][num_train : num_train + num_valid]
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

    with open(data_test, 'wb') as f:
        data = []
        features = cls_set[0][num_train+num_valid :]
        labels = cls_set[1][num_train+num_valid :]
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

def load():
    with open(data_train, 'rb') as f:
        data = np.load(f)
        print(data)


if __name__ == '__main__':
    #main()
    load()
