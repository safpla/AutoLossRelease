# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os, sys

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from dataio.dataset_mnist import Dataset_mnist
import utils
from models.basic_model import Basic_model

logger = utils.get_logger()

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Mnist(Basic_model):
    def __init__(self, config, exp_name='new_exp'):
        self.config = config
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)
        self.exp_name = exp_name
        train_data_file = config.data_dir
        valid_data_file = config.data_dir
        self.train_dataset = Dataset_mnist()
        self.train_dataset.load_mnist(train_data_file,
                                      mode=tf.estimator.ModeKeys.TRAIN)
        self.valid_dataset = Dataset_mnist()
        self.valid_dataset.load_mnist(valid_data_file,
                                      mode=tf.estimator.ModeKeys.EVAL)
        self.reset()
        self._build_placeholder()
        self._build_graph()

    def reset(self):
        pass

    def _build_placeholder(self):
        with self.graph.as_default():
            self.x_plh = tf.placeholder(shape=[None, 784], dtype=tf.float32)
            self.y_plh = tf.placeholder(shape=[None, 10], dtype=tf.float32)
            self.keep_prob = tf.placeholder(tf.float32)

    def _build_graph(self):
        with self.graph.as_default():
            x_image = tf.reshape(self.x_plh, [-1, 28, 28, 1])

            # First convolutional layer - maps one grayscale image to 32 feature maps.
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

            # Pooling layer - downsamples by 2X.
            h_pool1 = max_pool_2x2(h_conv1)

            # Second convolutional layer -- maps 32 feature maps to 64.
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

            # Second pooling layer.
            h_pool2 = max_pool_2x2(h_conv2)

            # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
            # is down to 7x7x64 feature maps -- maps this to 1024 features.
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # Dropout - controls the complexity of the model, prevents co-adaptation of
            # features.
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            # Map the 1024 features to 10 classes, one for each digit
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_plh,
                                                        logits=y_conv))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                                          tf.argmax(self.y_plh, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.accuracy = accuracy
            self.train_step = train_step
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.softmax = tf.nn.softmax(y_conv)

    def train(self):
        data = self.train_dataset.next_batch(self.config.batch_size)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y, self.keep_prob: 0.5}
        self.sess.run(self.train_step, feed_dict=feed_dict)

    def valid(self, dataset=None, fetch=None):
        """ test on validation set """
        if not dataset:
            dataset = self.valid_dataset
        data = dataset.next_batch(dataset.num_examples)
        x = data['input']
        y = data['target']

        feed_dict = {self.x_plh: x, self.y_plh: y, self.keep_prob: 1.0}
        if not fetch:
            fetch = [self.accuracy]
        result = self.sess.run(fetch, feed_dict=feed_dict)
        return result


if __name__ == '__main__':
    config_path = os.path.join(root_path, 'config/gan')
    config = utils.load_config(config_path)
    model = Mnist(config, exp_name='mnist_classification')
    model.initialize_weights()
    max_iteration = 20000
    for i in range(max_iteration):
        if i % 400 == 0:
            test_acc = model.valid()
            print('step: {}, validation accuracy: {}'.format(i, test_acc))
        model.train()
    test_acc = model.valid()
    print('test accuracy: {}'.format(test_acc))
    model.save_model(max_iteration)
