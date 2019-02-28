""" Generate toy dataset for linear classification with L1, L2 regularization """
# __author__ == 'Haowen Xu'
# __data__ == '04_25_2018'

import os
import sys
import json
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
import utils

import numpy as np
import sklearn.datasets as d

def save_to_file(features, labels, filename):
    with open(filename, 'wb') as f:
        data = []
        for feature, label in zip(features, labels):
            data.append({'x': feature, 'y': label})
        np.save(f, data)
        f.close()

def main():
    config = utils.load_config(os.path.join(root_path, 'config/cls'))
    dim = config.dim_input_task
    num_train_ctrl = config.num_sample_train_ctrl
    num_valid_ctrl = config.num_sample_valid_ctrl
    num_train_task = config.num_sample_train_task
    num_valid_task = config.num_sample_valid_task
    num_test = config.num_sample_test

    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(os.path.join(data_dir, config.task)):
        os.mkdir(os.path.join(data_dir, config.task))

    data_train_ctrl = os.path.join(data_dir, config.train_ctrl_data_file)
    data_valid_ctrl = os.path.join(data_dir, config.valid_ctrl_data_file)
    data_train_task = os.path.join(data_dir, config.train_task_data_file)
    data_valid_task = os.path.join(data_dir, config.valid_task_data_file)
    data_test = os.path.join(data_dir, config.test_data_file)
    np.random.seed(config.random_seed)
    n_samples = num_train_ctrl + num_valid_ctrl + \
                num_train_task + num_valid_task + \
                num_test
    cls_set = d.make_classification(n_samples=n_samples,
                                    n_features=dim,
                                    n_informative=2,
                                    n_clusters_per_class=2,
                                    n_redundant=2)
    features = cls_set[0]
    labels = cls_set[1]
    start = 0
    save_to_file(features[start:start+num_train_ctrl], 
        labels[start:start+num_train_ctrl], data_train_ctrl)
    start = start + num_train_ctrl
    save_to_file(features[start:start+num_valid_ctrl], 
        labels[start:start+num_valid_ctrl], data_valid_ctrl)
    start = start + num_valid_ctrl
    save_to_file(features[start:start+num_train_task], 
        labels[start:start+num_train_task], data_train_task)
    start = start + num_train_task
    save_to_file(features[start:start+num_valid_task], 
        labels[start:start+num_valid_task], data_valid_task)
    start = start + num_valid_task
    save_to_file(features[start:start+num_test], 
        labels[start:start+num_test], data_test)
    assert(start + num_test == features.shape[0])


def test_load():
    with open(data_train, 'rb') as f:
        data = np.load(f)
        print(data)


if __name__ == '__main__':
    main()
    #test_load()
