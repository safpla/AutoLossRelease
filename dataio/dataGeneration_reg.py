""" Generate toy dataset for linear regression with L1, L2 regularization """

# __author__ == 'Haowen Xu'
# __data__ == '04_07_2018'

import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
import numpy as np
import json
import utils

config_path = os.path.join(root_path, 'config/reg')
config = utils.load_config(config_path)
dim = config.dim_input_task

num_train_ctrl = config.num_sample_train_ctrl
num_valid_ctrl = config.num_sample_valid_ctrl
num_train_task = config.num_sample_train_task
num_valid_task = config.num_sample_valid_task
num_test = config.num_sample_test

mean_noise = config.mean_noise
var_noise = config.var_noise
data_dir = config.data_dir
data_train_ctrl = os.path.join(data_dir, config.train_ctrl_data_file)
data_valid_ctrl = os.path.join(data_dir, config.valid_ctrl_data_file)
data_train_task = os.path.join(data_dir, config.train_task_data_file)
data_valid_task = os.path.join(data_dir, config.valid_task_data_file)
data_test = os.path.join(data_dir, config.test_data_file)
np.random.seed(1)


def linear_func(x, w):
    return np.dot(x, w)

def save_to_file(w, filename, num):
    with open(filename, 'wb') as  f:
        data = []
        for i in range(num):
            x = 10 * (np.random.rand(dim) - 0.5)
            y = linear_func(x, w) + np.random.normal(loc=mean_noise,
                                                     scale=var_noise)
            data.append({'x': x, 'y': y})
        np.save(f, data)
        f.close()

def main():
    # generator training dataset
    w = np.random.rand(dim) - 0.5
    print('weight: ', w)
    save_to_file(w, data_train_ctrl, num_train_ctrl)
    save_to_file(w, data_valid_ctrl, num_valid_ctrl)
    save_to_file(w, data_train_task, num_train_task)
    save_to_file(w, data_valid_task, num_valid_task)
    save_to_file(w, data_test, num_test)
    #mse = 0
    #for d in data:
    #    x = d['x']
    #    y = d['y']
    #    y_ = linear_func(x, w)
    #    mse += np.square(y - y_)
    #mse /= num_of_data_points_valid
    #print('valid_loss:', mse)


if __name__ == '__main__':
    main()
