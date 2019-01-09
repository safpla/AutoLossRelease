""" Decide which loss to update by Reinforce Learning """
# __Author__ == "Haowen Xu"
# __Data__ == "04-07-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import math

import utils
from models.basic_model import Basic_model

logger = utils.get_logger()

class Controller(Basic_model):
    def __init__(self, config, exp_name='new_exp_ctrl'):
        self.config = config
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)
        self.exp_name = exp_name
        self._build_placeholder()
        self._build_graph()

    def _build_placeholder(self):
        config = self.config
        a = config.dim_action_rl
        s = config.dim_state_rl
        with self.graph.as_default():
            self.state_plh = tf.placeholder(shape=[None, s],
                                            dtype=tf.float32)
            self.reward_plh = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_plh = tf.placeholder(shape=[None, a], dtype=tf.int32)
            self.lr_plh = tf.placeholder(dtype=tf.float32)

    def _build_graph(self):
        config = self.config
        x_size = config.dim_state_rl
        h_size = config.dim_hidden_rl
        a_size = config.dim_action_rl
        lr = self.lr_plh
        with self.graph.as_default():
            model_name = config.controller_model_name
            initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            if model_name == '2layer':
                hidden = slim.fully_connected(self.state_plh, h_size,
                                              weights_initializer=initializer,
                                              activation_fn=tf.nn.leaky_relu)
                self.logits = slim.fully_connected(hidden, a_size,
                                                   weights_initializer=initializer,
                                                   activation_fn=None)
                self.output = tf.nn.softmax(self.logits)
            elif model_name == '2layer_logits_clipping':
                hidden = slim.fully_connected(self.state_plh, h_size,
                                              weights_initializer=initializer,
                                              activation_fn=tf.nn.leaky_relu)
                self.logits = slim.fully_connected(hidden, a_size,
                                                   weights_initializer=initializer,
                                                   activation_fn=None)
                self.output = tf.nn.softmax(self.logits /
                                            config.logit_clipping_c)
            elif model_name == 'linear':
                self.logits = slim.fully_connected(self.state_plh, a_size,
                                                   weights_initializer=initializer,
                                                   activation_fn=None)
                self.output = tf.nn.softmax(self.logits)
            elif model_name == 'linear_logits_clipping':
                #self.logits = slim.fully_connected(self.state_plh, a_size,
                #                                   weights_initializer=initializer,
                #                                   activation_fn=None)
                # ----Old version----
                w = tf.get_variable('w', shape=[x_size, a_size], dtype=tf.float32,
                                    initializer=initializer)
                b = tf.get_variable('b', shape=[a_size], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                self.logits = tf.matmul(self.state_plh, w) + b
                self.output = tf.nn.softmax(self.logits /
                                            config.logit_clipping_c)
            else:
                raise Exception('Invalid controller_model_name')

            self.chosen_action = tf.argmax(self.output, 1)
            self.action = tf.cast(tf.argmax(self.action_plh, 1), tf.int32)
            self.indexes = tf.range(0, tf.shape(self.output)[0])\
                * tf.shape(self.output)[1] + self.action
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                                self.indexes)
            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)
                                        * self.reward_plh)

            # ----Restore gradients and update them after several iterals.----
            if config.optimizer_ctrl == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            elif config.optimizer_ctrl == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            self.tvars = tf.trainable_variables()
            tvars = self.tvars
            self.gradient_plhs = []
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_plh')
                self.gradient_plhs.append(placeholder)

            gvs = optimizer.compute_gradients(self.loss, tvars)
            self.grads = [grad for grad, _ in gvs]
            self.train_op = optimizer.apply_gradients(zip(self.gradient_plhs, tvars))
            #self.train_op = optimizer.apply_gradients(gvs)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def sample(self, state, explore_rate=0, discrete=True):
        #
        # Sample an action from a given state, probabilistically

        # Args:
        #     state: shape = [dim_state_rl]
        #     explore_rate: explore rate

        # Returns:
        #     action: shape = [dim_action_rl]
        #
        sess = self.sess
        a_dist = sess.run(self.output, feed_dict={self.state_plh: [state]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)

        # ----Handcraft classifier----
        #  ---Hard version---
        #  Threshold varies in different tasks, which is related to the SNR
        #  of the data
        #  ------
        #if state[0] > 1.2:
        #    a = 1
        #else:
        #    a = 0
        #mse_loss = state[2]
        #l1_loss = state[3]
        #if (l1_loss - mse_loss) / l1_loss > 0.01:
        #    a = 1
        #else:
        #    a = 0
        #if state[4] > -0.19:
        #    a = 0
        #else:
        #    a = 1

        action = np.zeros(len(a_dist[0]), dtype='i')
        action[a] = 1
        if discrete:
            return action
        else:
            return a_dist[0]

    def train_one_step(self, gradBuffer, sh, ah, rh, lr):
        feed_dict = dict(zip(self.gradient_plhs, gradBuffer))
        feed_dict[self.lr_plh] = lr
        feed_dict[self.reward_plh] = rh
        feed_dict[self.action_plh] = ah
        feed_dict[self.state_plh] = sh

        self.sess.run(self.train_op, feed_dict=feed_dict)

    def print_weights(self):
        sess = self.sess
        with self.graph.as_default():
            tvars = sess.run(self.tvars)
            for idx, var in enumerate(tvars):
                logger.info('idx:{}, var:{}'.format(idx, var))

    def initialize_weights(self):
        self.sess.run(self.init)

    def get_weights(self):
        return self.sess.run(self.tvars)

    def get_gradients(self, state, action, reward):
        """ Return the gradients according to one episode

        Args:
            sess: Current tf.Session
            state: shape = [time_steps, dim_state_rl]
            action: shape = [time_steps, dim_action_rl]
            reward: shape = [time_steps]

        Returns:
            grads: Gradients of all trainable variables
        """
        sess = self.sess
        feed_dict = {self.reward_plh: reward,
                     self.action_plh: action,
                     self.state_plh: state,
                    }
        grads = sess.run(self.grads, feed_dict=feed_dict)
        return grads

