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

config_path = os.path.join(root_path, 'config/cls')
config = utils.load_config(config_path)
dim = config.dim_input_task
num_train_ctrl = config.num_sample_train_ctrl
num_valid_ctrl = config.num_sample_valid_ctrl
num_train_task = config.num_sample_train_task
num_valid_task = config.num_sample_valid_task
num_test = config.num_sample_test

data_train_ctrl = config.train_ctrl_data_file
data_valid_ctrl = config.valid_ctrl_data_file
data_train_task = config.train_task_data_file
data_valid_task = config.valid_task_data_file
data_test = config.test_data_file
np.random.seed(config.random_seed)


def linear_func(x, w):
    return np.dot(x, w)

def save_to_file(all_features, all_labels, filename, num):
    with open(filename, 'wb') as f:
        data = []
        features = all_features[:num]
        all_features = all_features[num:]
        labels = all_labels[:num]
        all_labels = all_labels[num:]
        for feature, label in zip(features, labels):
            data.append({'x': feature, 'y': label})
        np.save(f, data)
        f.close()

    return all_features, all_labels

def main():
    n_samples = num_train_ctrl + num_valid_ctrl +\
                num_train_task + num_valid_task +\
                num_test

    cls_set = d.make_classification(n_samples=n_samples,
                                    n_features=dim,
                                    n_informative=2,
                                    n_clusters_per_class=2,
                                    n_redundant=2)
    features = cls_set[0]
    labels = cls_set[1]
    folder = '/'.join(data_train_ctrl.split('/')[:-1])
    if not os.path.exists(folder):
        os.mkdir(folder)

    print(labels.shape)
    features, labels = save_to_file(features, labels, data_train_ctrl, num_train_ctrl)
    print(labels.shape)
    features, labels = save_to_file(features, labels, data_valid_ctrl, num_valid_ctrl)
    print(labels.shape)
    features, labels = save_to_file(features, labels, data_train_task, num_train_task)
    print(labels.shape)
    features, labels = save_to_file(features, labels, data_valid_task, num_valid_task)
    print(labels.shape)
    features, labels = save_to_file(features, labels, data_test, num_test)
    print(labels.shape)

def load():
    with open(data_train, 'rb') as f:
        data = np.load(f)
        print(data)


if __name__ == '__main__':
    main()
    #load()
