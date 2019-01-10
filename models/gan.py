""" This module implement a gan task """
# __Author__ == "Haowen Xu"
# __Data__ == "04-29-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os

from dataio.dataset_mnist import Dataset_mnist
import utils
from utils import inception_score_mnist
from utils import save_images
from models import layers
from models.basic_model import Basic_model
logger = utils.get_logger()

class Gan(Basic_model):
    '''
    Public variables (all task models should have these public variables):
        self.extra_info
        self.checkpoint_dir
        self.best_performance
        self.test_dataset
    '''
    def __init__(self, config, exp_name='new_exp_gan', arch=None):
        self.config = config
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)

        self.exp_name = exp_name

        self.arch = arch
        if arch:
            logger.info('architecture:')
            for key in sorted(arch.keys()):
                logger.info('{}: {}'.format(key, arch[key]))

        self.reset()
        self._load_datasets()
        self._build_placeholder()
        self._build_graph()

        self.reward_baseline = None
        # ema of metrics track over episodes
        n_steps = int(config.max_training_step / config.valid_frequency_task)
        self.metrics_track_baseline = -np.ones([n_steps])

        self.fixed_noise_128 = np.random.normal(size=(config.batch_size, config.dim_z))\
            .astype('float32')

    def _load_datasets(self):
        config = self.config
        self.train_dataset = Dataset_mnist()
        self.train_dataset.load_mnist(config.data_dir,
                                      tf.estimator.ModeKeys.TRAIN)
        self.valid_dataset = Dataset_mnist()
        self.valid_dataset.load_mnist(config.data_dir,
                                      tf.estimator.ModeKeys.EVAL)

    def reset(self):
        # ----Reset the model.----
        # TODO(haowen) The way to carry step number information should be
        # reconsiderd
        self.step_number = 0
        self.ema_gen_cost = None
        self.ema_disc_cost_real = None
        self.ema_disc_cost_fake = None
        self.prst_gen_cost = None
        self.prst_disc_cost_real = None
        self.prst_disc_cost_fake = None
        self.mag_gen_grad = None
        self.mag_disc_grad = None
        self.inception_score = 0
        self.checkpoint_dir = None

        # to control when to terminate the episode
        self.endurance = 0
        # The bigger the performance is, the better. In this case, performance
        # is the inception score. Naming it as performance in order to be
        # compatible with other tasks.
        self.best_performance = -1e10
        self.collapse = False
        self.previous_action = -1
        self.same_action_count = 0

    def _build_placeholder(self):
        with self.graph.as_default():
            dim_x = self.config.dim_x
            dim_z = self.config.dim_z
            self.real_data = tf.placeholder(tf.float32, shape=[None, dim_x],
                                            name='real_data')
            self.noise = tf.placeholder(tf.float32, shape=[None, dim_z],
                                        name='noise')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

    def _build_graph(self):
        dim_x = self.config.dim_x
        dim_z = self.config.dim_z
        batch_size = self.config.batch_size
        lr = self.config.lr_ctrl
        beta1 = self.config.beta1
        beta2 = self.config.beta2

        with self.graph.as_default():
            real_data = tf.cast(self.real_data, tf.float32)
            fake_data = self.generator(self.noise)
            disc_real = self.discriminator(real_data)
            disc_fake = self.discriminator(fake_data, reuse=True)

            gen_cost = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=disc_fake, labels=tf.ones_like(disc_fake)
                )
            )
            disc_cost_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=disc_fake, labels=tf.zeros_like(disc_fake)
                )
            )
            disc_cost_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=disc_real, labels=tf.ones_like(disc_real)
                )
            )
            disc_cost = (disc_cost_fake + disc_cost_real) / 2.

            tvars = tf.trainable_variables()
            gen_tvars = [v for v in tvars if 'Generator' in v.name]
            disc_tvars = [v for v in tvars if 'Discriminator' in v.name]

            gen_grad = tf.gradients(gen_cost, gen_tvars)
            disc_grad = tf.gradients(disc_cost, disc_tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                               beta1=beta1,
                                               beta2=beta2)
            gen_train_op = optimizer.apply_gradients(
                zip(gen_grad, gen_tvars))
            disc_train_op = optimizer.apply_gradients(
                zip(disc_grad, disc_tvars))

            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()
            self.fake_data = fake_data
            self.gen_train_op = gen_train_op
            self.disc_train_op = disc_train_op
            self.update = [gen_train_op, disc_train_op]

            self.gen_cost = gen_cost
            self.gen_grad = gen_grad
            self.gen_tvars = gen_tvars

            self.disc_cost_fake = disc_cost_fake
            self.disc_cost_real = disc_cost_real
            self.disc_grad = disc_grad
            self.disc_tvars = disc_tvars

            self.grads = gen_grad + disc_grad

    def generator(self, input):
        if self.arch:
            arch = self.arch
            dim_z = arch['dim_z']
            dim_c = arch['dim_c_G']
            if arch['activation_G'] == 'relu':
                activation_fn = tf.nn.relu
            elif arch['activation_G'] == 'leakyRelu':
                activation_fn = tf.nn.leaky_relu
            else:
                activation_fn = tf.nn.tanh
            batchnorm = arch['batchnorm_G']
        else:
            dim_z = self.config.dim_z
            dim_c = self.config.dim_c
            activation_fn = tf.nn.relu
            batchnorm = False

        with tf.variable_scope('Generator'):
            output = layers.linear(input, 7*7*2*dim_c, name='LN1')
            if batchnorm:
                output = layers.batchnorm(output, is_training=self.is_training,
                                        name='BN1')
            output = activation_fn(output)
            output = tf.reshape(output, [-1, 7, 7, 2*dim_c])

            output_shape = [-1, 14, 14, dim_c]
            output = layers.deconv2d(output, output_shape, name='Deconv2')
            if batchnorm:
                output = layers.batchnorm(output, is_training=self.is_training,
                                        name='BN2')
            output = activation_fn(output)

            output_shape = [-1, 28, 28, 1]
            output = layers.deconv2d(output, output_shape, name='Decovn3')
            output = tf.nn.tanh(output)

            return tf.reshape(output, [-1, 28*28])

    def discriminator(self, inputs, reuse=False):
        if self.arch:
            arch = self.arch
            dim_c = arch['dim_c_D']
            if arch['activation_D'] == 'relu':
                activation_fn = tf.nn.relu
            elif arch['activation_D'] == 'leakyRelu':
                activation_fn = tf.nn.leaky_relu
            else:
                activation_fn = tf.nn.tanh
            batchnorm = arch['batchnorm_D']
        else:
            dim_c = self.config.dim_c
            activation_fn = tf.nn.leaky_relu
            batchnorm = False

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 28, 28, 1])

            output = layers.conv2d(output, dim_c, name='Conv1')
            output = activation_fn(output)

            output = layers.conv2d(output, 2*dim_c, name='Conv2')
            if batchnorm:
                output = layers.batchnorm(output, is_training=self.is_training,
                                          name='BN2')
            output = activation_fn(output)

            output = tf.reshape(output, [-1, 7*7*2*dim_c])
            output = layers.linear(output, 1, name='LN3')

            return tf.reshape(output, [-1])

    def train(self, save_model=False):
        # TODO: haven't go through this part
        sess = self.sess
        config = self.config
        batch_size = config.batch_size
        dim_z = config.dim_z
        valid_frequency = config.valid_frequency_stud
        print_frequency = config.print_frequency_stud
        max_endurance = config.max_endurance_stud
        endurance = 0
        best_inps = 0
        inps_baseline = 0
        decay = config.metric_decay
        steps_per_iteration = config.disc_iters + config.gen_iters
        lrs = np.linspace(config.lr_start_stud,
                          config.lr_end_stud,
                          config.lr_decay_steps_stud)
        for step in range(config.max_training_step):
            lr = lrs[min(config.lr_decay_steps_stud-1, step)]
            if step % steps_per_iteration < config.disc_iters:
                # ----Update D network.----
                data = self.train_dataset.next_batch(batch_size)
                x = data['input']

                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.real_data: x,
                             self.is_training: True, self.lr: lr}
                sess.run(self.disc_train_op, feed_dict=feed_dict)
            else:
                # ----Update G network.----
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.is_training: True,
                             self.lr: lr}
                sess.run(self.gen_train_op, feed_dict=feed_dict)

            if step % valid_frequency == 0:
                logger.info('========Step{}========'.format(step))
                logger.info(endurance)
                inception_score = self.get_inception_score(config.inps_batches)
                logger.info(inception_score)
                if inps_baseline > 0:
                    inps_baseline = inps_baseline * decay \
                        + inception_score[0] * (1 - decay)
                else:
                    inps_baseline = inception_score[0]
                logger.info('inps_baseline: {}'.format(inps_baseline))
                self.generate_images(step)
                endurance += 1
                if inps_baseline > best_inps:
                    best_inps = inps_baseline
                    endurance = 0
                    if save_model:
                        self.save_model(step)

            if step % print_frequency == 0:
                data = self.train_dataset.next_batch(batch_size)
                x = data['input']
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.real_data: x,
                             self.is_training: False}
                fetch = [self.gen_cost,
                         self.disc_cost_fake,
                         self.disc_cost_real]
                r = sess.run(fetch, feed_dict=feed_dict)
                logger.info('gen_cost: {}'.format(r[0]))
                logger.info('disc_cost fake: {}, real: {}'.format(r[1], r[2]))

            if endurance > max_endurance:
                break
        logger.info('best_inps: {}'.format(best_inps))

    def response(self, action):
        """
         Given an action, return the new state, reward and whether dead

         Args:
             action: one hot encoding of actions

         Returns:
             state: shape = [dim_input_ctrl]
             reward: shape = [1]
             dead: boolean
        """
        # GANs only uses one dataset
        dataset = self.train_dataset

        sess = self.sess
        config = self.config
        batch_size = config.batch_size
        dim_z = config.dim_z
        alpha = config.state_decay

        data = self.train_dataset.next_batch(batch_size)
        x = data['input']
        z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
        feed_dict = {self.noise: z, self.real_data: x,
                     self.is_training: True}
        a = np.argmax(np.array(action))

        # ----To detect collapse.----
        if a == self.previous_action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0
        self.previous_action = a

        fetch = [self.update[a], self.gen_grad, self.disc_grad,
                self.gen_cost, self.disc_cost_real, self.disc_cost_fake]
        _, gen_grad, disc_grad, gen_cost, disc_cost_real, disc_cost_fake = \
            sess.run(fetch, feed_dict=feed_dict)

        self.mag_gen_grad = self.get_grads_magnitude(gen_grad)
        self.mag_disc_grad = self.get_grads_magnitude(disc_grad)
        self.prst_gen_cost = gen_cost
        self.prst_disc_cost_real = disc_cost_real
        self.prst_disc_cost_fake = disc_cost_fake

        # ----Update state.----
        self.step_number += 1
        if self.ema_gen_cost is None:
            self.ema_gen_cost = gen_cost
            self.ema_disc_cost_real = disc_cost_real
            self.ema_disc_cost_fake = disc_cost_fake
        else:
            self.ema_gen_cost = self.ema_gen_cost * alpha\
                + gen_cost * (1 - alpha)
            self.ema_disc_cost_real = self.ema_disc_cost_real * alpha\
                + disc_cost_real * (1 - alpha)
            self.ema_disc_cost_fake = self.ema_disc_cost_fake * alpha\
                + disc_cost_fake * (1 - alpha)

        self.extra_info = {'gen_cost': self.ema_gen_cost,
                           'disc_cost_real': self.ema_disc_cost_real,
                           'disc_cost_fake': self.ema_disc_cost_fake}

        reward = 0
        # ----Early stop and record best result.----
        dead = self.check_terminate()
        state = self.get_state()
        return state, reward, dead

    def update_inception_score(self, score):
        self.best_performance = score

    def get_state(self):
        if self.step_number == 0:
            state = [0] * self.config.dim_input_ctrl
        else:
            state = [
                     math.log(self.mag_disc_grad / self.mag_gen_grad),
                     self.ema_gen_cost - 1,
                     (self.ema_disc_cost_real + self.ema_disc_cost_fake) / 2,
                     self.inception_score - 8,
                    ]
        return np.array(state, dtype='f')

    def check_terminate(self):
        # TODO(haowen)
        # Early stop and recording the best result
        # Episode terminates on two condition:
        # 1) Convergence: inception score doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action
        if self.same_action_count > 500:
            logger.info('Terminate reason: Collapse')
            self.collapse = True
            return True
        step = self.step_number
        if step % self.config.valid_frequency_task == 0:
            self.endurance += 1
            inception_score = self.get_inception_score(self.config.inps_batches)
            inps = inception_score[0]
            self.inception_score = inps
            logger.info('Step: {}, inps: {}'.format(step, inps))
            #decay = self.config.metric_decay
            #if self.inps_ema > 0:
            #    self.inps_ema = self.inps_ema * decay + inps * (1 - decay)
            #else:
            #    self.inps_ema = inps
            if self.inception_score > self.best_performance:
                self.best_step = self.step_number
                self.best_performance = self.inception_score
                self.endurance = 0
                self.save_model(step, mute=True)

        if step > self.config.max_training_step:
            return True
        if self.config.stop_strategy_task == 'exceeding_endurance' and \
                self.endurance > self.config.max_endurance_task:
            return True
        return False

    def get_step_reward(self, step):
        step = int(step / self.config.print_frequency_ctrl) - 1
        inps = self.inception_score
        reward = self.config.reward_c * inps ** 2
        if self.metrics_track_baseline[step] == -1:
            self.metrics_track_baseline[step] = inps
            #logger.info(self.metrics_track_baseline)
            return 0
        baseline_inps = self.metrics_track_baseline[step]
        baseline_reward = self.config.reward_c * baseline_inps ** 2
        adv = reward - baseline_reward
        adv = min(adv, self.config.reward_max_value)
        adv = max(adv, -self.config.reward_max_value)
        decay = self.config.inps_baseline_decay
        self.metrics_track_baseline[step] = \
            decay * self.metrics_track_baseline[step] + (1 - decay) * inps
        #logger.info(self.metrics_track_baseline)
        return adv

    def get_final_reward(self):
        if self.collapse:
            return 0, -self.config.reward_max_value
        inps = self.best_performance
        reward = self.config.reward_c * inps ** 2

        # Calculate baseline
        if self.reward_baseline is None:
            self.reward_baseline = reward
        else:
            decay = self.config.reward_baseline_decay
            self.reward_baseline = decay * self.reward_baseline\
                + (1 - decay) * reward

        # Calcuate advantage
        adv = reward - self.reward_baseline
        # reward_min_value < abs(adv) < reward_max_value
        adv = math.copysign(max(self.config.reward_min_value, abs(adv)), adv)
        adv = math.copysign(min(self.config.reward_max_value, abs(adv)), adv)
        return reward, adv

    def get_inception_score(self, num_batches, splits=None):
        all_samples = []
        config = self.config
        if not splits:
            splits = config.inps_splits
        batch_size = 100
        dim_z = config.dim_z
        for i in range(num_batches):
            z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
            feed_dict = {self.noise: z, self.is_training: False}
            samples = self.sess.run(self.fake_data, feed_dict=feed_dict)
            all_samples.append(samples)
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = all_samples.reshape((-1, 28*28))
        return inception_score_mnist.get_inception_score(all_samples,
                                                   splits=splits)

    def generate_images(self, step):
        feed_dict = {self.noise: self.fixed_noise_128,
                     self.is_training: False}
        samples = self.sess.run(self.fake_data, feed_dict=feed_dict)
        checkpoint_dir = os.path.join(self.config.save_images_dir, self.exp_name)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, 'images_{}.jpg'.format(step))
        save_images.save_images(samples.reshape((-1, 28, 28, 1)), save_path)

