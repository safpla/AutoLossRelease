""" This module implement a toy task: linear classification """
# __Author__ == "Haowen Xu"
# __Data__ == "04-25-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math

from dataio.dataset import Dataset
import utils
from models.basic_model import Basic_model
logger = utils.get_logger()


def weight_variable(shape, name='b'):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name='b'):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Cls(Basic_model):
    '''
    Public variables (all task models should have these public variables):
        self.extra_info
        self.checkpoint_dir
        self.best_performance
        self.test_dataset
    '''

    def __init__(self, config, exp_name='new_exp'):
        self.config = config
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)
        self.exp_name = exp_name

        self.reset()
        self._load_datasets()
        self._build_placeholder()
        self._build_graph()
        self.reward_baseline = None # average reward over episodes
        self.improve_baseline = None # averge improvement over steps

    def _load_datasets(self):
        config = self.config
        train_ctrl_data_file = os.path.join(config.data_dir, config.train_ctrl_data_file)
        train_task_data_file = os.path.join(config.data_dir, config.train_task_data_file)
        valid_ctrl_data_file = os.path.join(config.data_dir, config.valid_ctrl_data_file)
        valid_task_data_file = os.path.join(config.data_dir, config.valid_task_data_file)
        test_data_file = os.path.join(config.data_dir, config.test_data_file)
        self.train_ctrl_dataset = Dataset()
        self.train_ctrl_dataset.load_npy(train_ctrl_data_file)
        self.valid_ctrl_dataset = Dataset()
        self.valid_ctrl_dataset.load_npy(valid_ctrl_data_file)
        self.train_task_dataset = Dataset()
        self.train_task_dataset.load_npy(train_task_data_file)
        self.valid_task_dataset = Dataset()
        self.valid_task_dataset.load_npy(valid_task_data_file)
        self.test_dataset = Dataset()
        self.test_dataset.load_npy(test_data_file)

    def reset(self):
        # ----Reset the model.----
        # TODO(haowen) The way to carry step number information should be
        # reconsiderd
        self.step_number = [0]
        self.previous_ce_loss = [0] * self.config.num_pre_loss
        self.previous_l1_loss = [0] * self.config.num_pre_loss
        self.previous_valid_acc = [0] * self.config.num_pre_loss
        self.previous_train_acc = [0] * self.config.num_pre_loss
        self.previous_valid_loss = [0] * self.config.num_pre_loss
        self.previous_train_loss = [0] * self.config.num_pre_loss
        self.checkpoint_dir = None

        # to control when to terminate the episode
        self.endurance = 0
        # The bigger the performance is, the better. In this case, performance
        # is accuracy. Naming it as performance in order to be compatible with
        # other tasks.
        self.best_performance = -1e10
        self.improve_baseline = None

    def _build_placeholder(self):
        x_size = self.config.dim_input_task
        with self.graph.as_default():
            self.x_plh = tf.placeholder(shape=[None, x_size], dtype=tf.float32)
            self.y_plh = tf.placeholder(shape=[None], dtype=tf.int32)

    def _build_graph(self):
        x_size = self.config.dim_input_task
        h_size = self.config.dim_hidden_task
        y_size = self.config.dim_output_task
        lr = self.config.lr_task

        with self.graph.as_default():
            w1 = weight_variable([x_size, h_size], name='w1')
            b1 = bias_variable([h_size], name='b1')
            hidden = tf.nn.relu(tf.matmul(self.x_plh, w1) + b1)

            w2 = weight_variable([h_size, y_size], name='w2')
            b2 = bias_variable([y_size], name='b2')
            self.pred = tf.matmul(hidden, w2) + b2

            # define loss
            # cross entropy loss
            self.loss_ce = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_plh,
                    logits=self.pred,
                    name='loss')
            )
            y_ = tf.argmax(self.pred, 1, output_type=tf.int32)
            correct_prediction = tf.equal(y_, self.y_plh)
            self.correct_prediction = correct_prediction
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                   tf.float32))

            tvars = tf.trainable_variables()
            self.tvars = tvars
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=self.config.lambda_task, scope=None)
            self.loss_l1 = tf.contrib.layers.apply_regularization(
                l1_regularizer, tvars)
            self.loss_total = self.loss_ce + self.loss_l1

            # ----Define update operation.----
            self.update_ce = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_ce)
            self.update_l1 = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_l1)
            self.update_total = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_total)
            self.update = [self.update_ce, self.update_l1]
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def valid(self, dataset=None):
        if not dataset:
            if self.config.args.task_mode == 'train':
                dataset = self.valid_ctrl_dataset
            elif self.config.args.task_mode == 'test':
                dataset = self.valid_task_dataset
            elif self.config.args.task_mode == 'baseline':
                dataset = self.valid_task_dataset
            else:
                logger.exception('Unexcepted task_mode: {}'.\
                                 format(self.config.args.task_mode))
        data = dataset.next_batch(dataset.num_examples)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_ce, self.accuracy, self.pred, self.y_plh]
        [loss_ce, acc, pred, gdth] = self.sess.run(fetch, feed_dict=feed_dict)
        return loss_ce, acc, pred, gdth

    def response(self, action, lr, mode='TRAIN'):
        """ Given an action, return the new state, reward and whether dead

        Args:
            action: one hot encoding of actions

        Returns:
            state: shape = [dim_input_ctrl]
            reward: shape = [1]
            dead: boolean
        """
        if self.config.args.task_mode == 'train':
            dataset = self.train_ctrl_dataset
        elif self.config.args.task_mode == 'test':
            dataset = self.train_task_dataset
        elif self.config.args.task_mode == 'baseline':
            dataset = self.train_task_dataset
        else:
            logger.exception('Unexcepted mode: {}'.\
                             format(self.config.args.task_mode))
        data = dataset.next_batch(self.config.batch_size)

        sess = self.sess
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}

        a = np.argmax(np.array(action))
        sess.run(self.update[a], feed_dict=feed_dict)

        fetch = [self.loss_ce, self.loss_l1, self.accuracy]
        loss_ce, loss_l1, acc = sess.run(fetch, feed_dict=feed_dict)
        valid_loss, valid_acc, _, _ = self.valid()
        train_loss, train_acc = loss_ce, acc

        # ----Update state.----
        self.previous_ce_loss = self.previous_ce_loss[1:] + [loss_ce.tolist()]
        self.previous_l1_loss = self.previous_l1_loss[1:] + [loss_l1.tolist()]
        self.step_number[0] += 1
        self.previous_valid_loss = self.previous_valid_loss[1:]\
            + [valid_loss.tolist()]
        self.previous_train_loss = self.previous_train_loss[1:]\
            + [train_loss.tolist()]
        self.previous_valid_acc = self.previous_valid_acc[1:]\
            + [valid_acc.tolist()]
        self.previous_train_acc = self.previous_train_acc[1:]\
            + [train_acc.tolist()]

        # Public variable
        self.extra_info = {'valid_loss': valid_loss,
                           'train_loss': train_loss,
                           'valid_acc': valid_acc,
                           'train_acc': train_acc}

        reward = self.get_step_reward()
        # ----Early stop and record best result.----
        dead = self.check_terminate()
        state = self.get_state()
        return state, reward, dead

    def check_terminate(self):
        # TODO(haowen)
        # Early stop and recording the best result
        # Episode terminates on two condition:
        # 1) Convergence: valid loss doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action (not implement yet)
        step = self.step_number[0]
        if step % self.config.valid_frequency_task == 0:
            self.endurance += 1
            loss, acc, _, _ = self.valid()
            if acc > self.best_performance:
                self.best_step = self.step_number[0]
                self.best_performance = acc
                self.endurance = 0
                if not self.config.args.task_mode == 'train':
                    self.save_model(step, mute=True)

        if step > self.config.max_training_step:
            return True
        if self.config.stop_strategy_task == 'exceeding_endurance' and \
                self.endurance > self.config.max_endurance_task:
            return True
        return False

    def get_step_reward(self):
        # TODO(haowen) Use the decrease of validation loss as step reward
        if self.improve_baseline is None:
            # ----First step, nothing to comparing with.----
            improve = 0.1
        else:
            improve = (self.previous_valid_loss[-2] - self.previous_valid_loss[-1])

        # TODO(haowen) Try to use sqrt function instead of sign function
        # ----With baseline.----
        if self.improve_baseline is None:
            self.improve_baseline = improve
        decay = self.config.reward_baseline_decay
        self.improve_baseline = decay * self.improve_baseline\
            + (1 - decay) * improve

        value = math.sqrt(abs(improve) / (abs(self.improve_baseline) + 1e-5))
        #value = abs(improve) / (abs(self.improve_baseline) + 1e-5)
        value = min(value, self.config.reward_max_value)
        return math.copysign(value, improve) * self.config.reward_step_ctrl

    def get_final_reward(self):
        '''
        Return:
            reward: real reward
            adv: advantage, subtract the baseline from reward and then normalize it
        '''
        assert self.best_performance > -1e10 + 1
        acc = max(self.best_performance, 1 / self.config.dim_output_task)
        reward = -self.config.reward_c / acc

        # Calculate baseline
        if self.reward_baseline is None:
            self.reward_baseline = reward
        else:
            decay = self.config.reward_baseline_decay
            self.reward_baseline = decay * self.reward_baseline\
                + (1 - decay) * reward

        # Calculate advantage
        adv = reward - self.reward_baseline
        adv = min(adv, self.config.reward_max_value)
        adv = max(adv, -self.config.reward_max_value)
        return reward, adv

    def get_state(self):
        abs_diff = []
        rel_diff = []
        if self.improve_baseline is None:
            ib = 1
        else:
            ib = self.improve_baseline


        # relative difference between valid_loss and train_loss
        v = self.previous_valid_loss[-1]
        t = self.previous_train_loss[-1]
        abs_diff = v - t
        if t > 1e-6:
            rel_diff = (v - t) / t
        else:
            rel_diff = 0
        state0 = rel_diff

        # normalized baseline improvement
        state1 = 1 + math.log(abs(ib) + 1e-5) / 12

        # normalized mse ce_loss and l1_loss:
        ce_loss = self.previous_ce_loss[-1]
        state2 = 1 + math.log(ce_loss + 1e-5) / 12

        l1_loss = self.previous_l1_loss[-1]
        state3 = 1 + math.log(l1_loss + 1e-5) / 12

        # train_acc, valid_acc and their difference
        state4 = self.previous_train_acc[-1]
        state5 = self.previous_valid_acc[-1]
        state6 = state4 - state5

        state = [state0, state1, state2, state3, state4, state5, state6]
        return np.array(state, dtype='f')

