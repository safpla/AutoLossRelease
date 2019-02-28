""" Generate toy dataset for linear regression with L1, L2 regularization """
# __author__ == 'Haowen Xu'
# __data__ == '04_07_2018'

import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import numpy as np
import json
import utils


def linear_func(x, w):
    return np.dot(x, w)

def save_to_file(data, filename):
    with open(filename, 'wb') as  f:
        np.save(f, data)
        f.close()

def gen_data(w, dim, num, mean_noise, var_noise):
    data = []
    for i in range(num):
        x = 10 * (np.random.rand(dim) - 0.5)
        y = linear_func(x, w) + np.random.normal(loc=mean_noise,
                                                 scale=var_noise)
        data.append({'x': x, 'y': y})
    return data
        
def cal_baseline_mse(data):
    mse = 0
    cnt = 0
    for d in data:
        x = d['x']
        y = d['y']
        y_ = linear_func(x, w)
        mse += np.square(y - y_)
        cnt += 1
    mse /= cnt 
    print('valid_loss:', mse)

def main():
    # generator training dataset
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
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(os.path.join(data_dir, config.task)):
        os.mkdir(os.path.join(data_dir, config.task))
    data_train_ctrl = os.path.join(data_dir, config.train_ctrl_data_file)
    data_valid_ctrl = os.path.join(data_dir, config.valid_ctrl_data_file)
    data_train_task = os.path.join(data_dir, config.train_task_data_file)
    data_valid_task = os.path.join(data_dir, config.valid_task_data_file)
    data_test = os.path.join(data_dir, config.test_data_file)

    np.random.seed(1)
    w = np.random.rand(dim) - 0.5
    data = gen_data(w, dim, num_train_ctrl, mean_noise, var_noise)
    save_to_file(data, data_train_ctrl)
    data = gen_data(w, dim, num_valid_ctrl, mean_noise, var_noise)
    save_to_file(data, data_valid_ctrl)
    data = gen_data(w, dim, num_train_task, mean_noise, var_noise)
    save_to_file(data, data_train_task)
    data = gen_data(w, dim, num_valid_task, mean_noise, var_noise)
    save_to_file(data, data_valid_task)
    data = gen_data(w, dim, num_test, mean_noise, var_noise)
    save_to_file(data, data_test)


if __name__ == '__main__':
    main()
