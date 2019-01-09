""" Generate toy dataset for linear regression with L1, L2 regularization """

# __author__ == 'Haowen Xu'
# __data__ == '04_07_2018'

import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
import numpy as np
import json
import utils

config_path = os.path.join(root_path, 'config/regression.cfg')
config = utils.Parser(config_path)
dim = config.dim_input_stud
num_of_data_points_train = config.num_sample_train
num_of_data_points_valid = config.num_sample_valid
num_of_data_points_train_stud = config.num_sample_train_stud
mean_noise = config.mean_noise
var_noise = config.var_noise
data_train = config.train_data_file
data_valid = config.valid_data_file
data_train_stud = config.train_stud_data_file
np.random.seed(1)


def linear_func(x, w):
    return np.dot(x, w)

def main():
    # generator training dataset
    w = np.random.rand(dim) - 0.5
    #w = np.array([0.1, 0.5, 0.5, -0.1, 0.2, -0.2, 0.3, 0.1])
    print('weight: ', w)

    with open(data_train, 'wb') as f:
        data = []
        for i in range(num_of_data_points_train):
            x = 10 * (np.random.rand(dim) - 0.5)
            y = linear_func(x, w) + np.random.normal(loc=mean_noise,
                                                     scale=var_noise)
            data.append({'x': x, 'y': y})
        print(data)
        np.save(f, data)
        f.close()

    with open(data_valid, 'wb') as f:
        data = []
        for i in range(num_of_data_points_valid):
            x = 10 * (np.random.rand(dim) - 0.5)
            y = linear_func(x, w) + np.random.normal(loc=mean_noise,
                                                     scale=var_noise)
            data.append({'x': x, 'y': y})
        np.save(f, data)
        f.close()

    with open(data_train_stud, 'wb') as f:
        data = []
        for i in range(num_of_data_points_train_stud):
            x = 10 * (np.random.rand(dim) - 0.5)
            y = linear_func(x, w) + np.random.normal(loc=mean_noise,
                                                     scale=var_noise)
            data.append({'x': x, 'y': y})
        print(data)
        np.save(f, data)
        f.close()

    mse = 0
    for d in data:
        x = d['x']
        y = d['y']
        y_ = linear_func(x, w)
        mse += np.square(y - y_)
    mse /= num_of_data_points_valid
    print('valid_loss:', mse)

def load():
    with open(data_train, 'rb') as f:
        data = np.load(f)
        print(data)


if __name__ == '__main__':
    main()
