# __Author__ == 'Haowen Xu'
# __Data__ == '05-04-2018'

import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
import numpy as np
import utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

config_path = os.path.join(root_path, 'config/gan_grid.cfg')
config = utils.Parser(config_path)
dim = config.dim_input_stud
num_train = config.num_sample_train
num_valid = config.num_sample_valid
num_test = config.num_sample_test
var_noise = config.var_noise
data_train = config.train_data_file
data_valid = config.valid_data_file
data_test = config.test_data_file
np.random.seed(1)


def plot_data(all_data):
    plt.scatter(all_data[:,0], all_data[:,1])
    plt.show()
    plt.savefig('grid_distribution.png', bbox_inches='tight')
    plt.close()

def main():
    n_samples = num_train + num_valid + num_test
    all_data = []
    #center = [-0.5, 0.5]
    center = [-1.0, -0.5, 0, 0.5, 1.0]
    dim = len(center)
    for i in range(dim):
        for j in range(dim):
            n_sample = int(n_samples / dim**2)
            data = np.random.normal(loc=0, scale=var_noise, size=(n_sample, 2))
            c = np.array([center[i], center[j]])
            all_data.append(data + c)
    all_data = np.concatenate(all_data)
    np.random.shuffle(all_data)
    plot_data(all_data)

    with open(data_train, 'wb') as f:
        data = []
        features = all_data[:num_train]
        labels = np.zeros([num_train])
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

    with open(data_valid, 'wb') as f:
        data = []
        features = all_data[num_train : num_train + num_valid]
        labels = np.zeros([num_valid])
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()

    with open(data_test, 'wb') as f:
        data = []
        features = all_data[num_train + num_valid :]
        labels = np.zeros([num_test])
        for feature, lable in zip(features, labels):
            data.append({'x': feature, 'y': lable})
        np.save(f, data)
        f.close()


if __name__ == '__main__':
    main()
