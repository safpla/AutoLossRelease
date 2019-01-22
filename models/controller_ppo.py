import tensorflow as tf
import numpy as np
import os
import random

from models.basic_model import Basic_model
import utils
logger = utils.get_logger()

class BasePPO(Basic_model):
    def __init__(self, config, exp_name):
        super(BasePPO, self).__init__(config, exp_name)
        self.update_steps = 0

    def _build_placeholder(self):
        '''
        Must include:
            self.state,
            self.action,
            self.reward,
            self.target_value,
            self.lr


        '''
        raise NotImplementedError

    def _build_graph(self):
        '''
        We use scope_names instead of Graph to manage variables
        '''
        pi, pi_param = self.build_actor_net('actor_net', trainable=True)
        old_pi, old_pi_param = self.build_actor_net('old_actor_net',
                                                    trainable=False)
        pi = pi + 1e-8
        old_pi = old_pi + 1e-8
        value, critic_param = self.build_critic_net('value_net')
        a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32),
                              self.action], axis=1)
        pi_wrt_a = tf.gather_nd(params=pi, indices=a_indices, name='pi_wrt_a')
        old_pi_wrt_a = tf.gather_nd(params=old_pi, indices=a_indices,
                                    name='old_pi_wrt_a')

        cliprange = self.config.cliprange_ctrl
        gamma = self.config.gamma_ctrl

        adv = self.target_value - value
        # NOTE: Stop passing gradient through adv
        adv = tf.stop_gradient(adv, name='critic_adv_stop_gradient')
        with tf.variable_scope('critic_loss'):
            critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x))
                                              for x in critic_param])
            mse_loss = tf.reduce_mean(tf.square(self.target_value - value))
            reg_param = 0.0
            self.critic_loss = mse_loss + reg_param * critic_reg_loss

        adv = self.target_value - value
        # NOTE: Stop passing gradient through adv
        adv = tf.stop_gradient(adv, name='actor_adv_stop_gradient')
        with tf.variable_scope('actor_loss'):
            ratio = pi_wrt_a / old_pi_wrt_a
            pg_losses1 = adv * ratio
            pg_losses2 = adv * tf.clip_by_value(ratio,
                                                1.0 - cliprange,
                                                1.0 + cliprange)
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(pi * tf.log(pi), 1))
            actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x))
                                            for x in pi_param])
            pg_loss = -tf.reduce_mean(tf.minimum(pg_losses1, pg_losses2))
            beta = self.config.entropy_bonus_beta_ctrl
            reg_param = 0.0
            self.actor_loss = pg_loss + beta * entropy_loss +\
                reg_param * actor_reg_loss

        self.sync_op = [tf.assign(oldp, p)
                        for oldp, p in zip(old_pi_param, pi_param)]
        optimizer1 = tf.train.AdamOptimizer(self.lr)
        optimizer2 = tf.train.AdamOptimizer(self.lr)
        self.train_op_actor = optimizer1.minimize(self.actor_loss)
        self.train_op_critic = optimizer2.minimize(self.critic_loss)

        self.tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope=self.exp_name)
        self.init = tf.variables_initializer(self.tvars)
        self.saver = tf.train.Saver(var_list=self.tvars, max_to_keep=1)

        self.pi = pi
        self.value = value
        self.ratio = ratio
        self.old_pi_param = old_pi_param

    def build_actor_net(self, scope, trainable):
        raise NotImplementedError

    def sample(self, states, epsilon=0.5):
        dim_a = self.config.dim_output_ctrl
        action = np.zeros(dim_a, dtype='i')
        if random.random() < epsilon:
            a = random.randint(0, dim_a - 1)
            action[a] = 1
            return action, 'random'
        else:
            pi = self.sess.run(self.pi, {self.state: states})[0]
            a = np.argmax(pi)
            #a = np.random.choice(dim_a, 1, p=pi)[0]
            action[a] = 1
            return action, pi

    def sync_net(self):
        self.sess.run(self.sync_op)
        #logger.info('{}: target_network synchronized'.format(self.exp_name))

    def update_critic(self, transition_batch, lr=0.001):
        state = transition_batch['state']
        action = transition_batch['action']
        reward = transition_batch['reward']
        target_value = transition_batch['target_value']
        fetch = [self.train_op_critic]
        feed_dict = {self.state: state,
                     self.action: action,
                     self.reward: reward,
                     self.target_value: target_value,
                     self.lr: lr}
        _ = self.sess.run(fetch, feed_dict)
        #tvar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        #                         '{}/value_net/'.format(self.exp_name))
        #self.print_weights(tvar)
        #if self.update_steps % 50 == 0:
        #    for i in range(5):
        #        print(check_value[i], target_value[i])

    def update_actor(self, transition_batch, lr=0.001):
        self.update_steps += 1
        state = transition_batch['state']
        action = transition_batch['action']
        reward = transition_batch['reward']
        target_value = transition_batch['target_value']

        fetch = [self.train_op_actor]
        feed_dict = {self.state: state,
                     self.action: action,
                     self.reward: reward,
                     self.target_value: target_value,
                     self.lr: lr}
        self.sess.run(fetch, feed_dict)

    def train_one_step(self, transition_batch, lr=0.001):
        self.update_actor(transition_batch, lr)
        self.update_critic(transition_batch, lr)

    def get_value(self, state):
        feed_dict = {self.state: state}
        value = self.sess.run(self.value, feed_dict)
        return value

    def print_weights(self):
        tvars = sess.run(self.tvars)
        for idx, var in enumerate(tvars):
            logger.info('idx: {}, var: {}'.format(idx, var))

    def get_weights(self):
        return self.sess.run(self.tvars)


class MlpPPO(BasePPO):
    def __init__(self, config, exp_name='MlpPPO'):
        super(MlpPPO, self).__init__(config, exp_name)
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()

    def _build_placeholder(self):
        config = self.config
        dim_s = config.dim_input_ctrl
        with tf.variable_scope('placeholder'):
            self.state = tf.placeholder(tf.float32,
                                        shape=[None, dim_s],
                                        name='state')
            self.action = tf.placeholder(tf.int32, shape=[None],
                                         name='action')
            self.reward = tf.placeholder(tf.float32, shape=[None],
                                         name='reward')
            self.target_value = tf.placeholder(tf.float32,
                                               shape=[None],
                                               name='target_value')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def build_actor_net(self, scope, trainable):
        with tf.variable_scope(scope):
            dim_h = self.config.dim_hidden_ctrl
            dim_a = self.config.dim_output_ctrl
            hidden = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=dim_h,
                activation_fn=tf.nn.tanh,
                trainable=trainable,
                scope='fc1')

            logits = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=dim_a,
                activation_fn=None,
                trainable=trainable,
                scope='fc2')

            output = tf.nn.softmax(logits)

            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))
            return output, param

    def build_critic_net(self, scope):
        with tf.variable_scope(scope):
            dim_h = self.config.dim_hidden_ctrl
            hidden = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=dim_h,
                activation_fn=tf.nn.tanh,
                scope='fc1')

            value = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=None,
                scope='fc2')
            value = tf.squeeze(value)

            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))
            return value, param
