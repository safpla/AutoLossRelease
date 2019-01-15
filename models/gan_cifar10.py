""" This module implement a gan task """
# __Author__ == "Haowen Xu"
# __Data__ == "04-29-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os

from dataio.dataset_cifar10 import Dataset_cifar10
import utils
from utils import inception_score
from utils import save_images
from models import layers
from models.gan import Gan
logger = utils.get_logger()

class Gan_cifar10(Gan):
    '''
    Public variables (all task models should have these public variables):
        self.extra_info
        self.checkpoint_dir
        self.best_performance
        self.test_dataset
    '''

    def __init__(self, config, exp_name='new_exp_gan_cifar10', arch=None):
        self.config = config
        self.graph = tf.Graph()
        self.exp_name = exp_name
        self.arch = arch
        if arch:
            logger.info('architecture:')
            for key in sorted(arch.keys()):
                logger.info('{}: {}'.format(key, arch[key]))
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)
        self.reset()
        self._load_datasets()

        self._build_placeholder()
        self._build_graph()

        self.reward_baseline = None
        # ema of metrics track over episodes
        n_steps = int(config.max_training_step / config.print_frequency_task)
        self.metrics_track_baseline = -np.ones([n_steps])
        self.fixed_noise_128 = np.random.normal(size=(128, config.dim_z))\
            .astype('float32')

    def _load_datasets(self):
        config = self.config
        self.train_dataset = Dataset_cifar10()
        self.train_dataset.load_cifar10(config.data_dir,
                                        tf.estimator.ModeKeys.TRAIN)
        self.valid_dataset = Dataset_cifar10()
        self.valid_dataset.load_cifar10(config.data_dir,
                                        tf.estimator.ModeKeys.EVAL)

    def _build_placeholder(self):
        with self.graph.as_default():
            dim_x = self.config.dim_x
            dim_z = self.config.dim_z
            self.real_data = tf.placeholder(tf.int32, shape=[None, dim_x],
                                                name='real_data')
            self.noise = tf.placeholder(tf.float32, shape=[None, dim_z],
                                        name='noise')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

    def _build_graph(self):
        dim_x = self.config.dim_x
        dim_z = self.config.dim_z
        batch_size = self.config.batch_size
        lr = self.config.lr_task
        beta1 = self.config.beta1
        beta2 = self.config.beta2

        with self.graph.as_default():
            real_data_t = 2*((tf.cast(self.real_data, tf.float32)/255.)-.5)
            real_data_NCHW = tf.reshape(real_data_t, [-1, 3, 32, 32])
            real_data_NHWC = tf.transpose(real_data_NCHW, perm=[0, 2, 3, 1])
            real_data = tf.reshape(real_data_NHWC, [-1, dim_x])
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
            batchnorm = True

        with tf.variable_scope('Generator'):
            output = layers.linear(input, 4*4*4*dim_c, name='LN1')
            if batchnorm:
                output = layers.batchnorm(output, is_training=self.is_training,
                                        name='BN1')
            output = activation_fn(output)
            output = tf.reshape(output, [-1, 4, 4, 4*dim_c])

            output_shape = [-1, 8, 8, 2*dim_c]
            output = layers.deconv2d(output, output_shape, name='Deconv2')
            if batchnorm:
                output = layers.batchnorm(output, is_training=self.is_training,
                                        name='BN2')
            output = activation_fn(output)

            output_shape = [-1, 16, 16, dim_c]
            output = layers.deconv2d(output, output_shape, name='Decovn3')
            if batchnorm:
                output = layers.batchnorm(output, is_training=self.is_training,
                                        name='BN3')
            output = activation_fn(output)

            output_shape = [-1, 32, 32, 3]
            output = layers.deconv2d(output, output_shape, name='Deconv4')
            output = tf.nn.tanh(output)

            return tf.reshape(output, [-1, 32*32*3])

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
            batchnorm = True

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 32, 32, 3])

            output = layers.conv2d(output, dim_c, name='Conv1')
            output = activation_fn(output)

            output = layers.conv2d(output, 2*dim_c, name='Conv2')
            if batchnorm:
                output = layers.batchnorm(output, is_training=self.is_training,
                                        name='BN2')
            output = activation_fn(output)

            output = layers.conv2d(output, 4*dim_c, name='Conv3')
            if batchnorm:
                output = layers.batchnorm(output, is_training=self.is_training,
                                        name='BN3')
            output = activation_fn(output)

            output = tf.reshape(output, [-1, 4*4*4*dim_c])
            output = layers.linear(output, 1, name='LN4')

            return tf.reshape(output, [-1])

    def get_state(self):
        if self.step_number == 0:
            state = [0] * self.config.dim_output_ctrl
        else:
            state = [
                     math.log(self.mag_disc_grad / self.mag_gen_grad),
                     self.ema_gen_cost - 1,
                     (self.ema_disc_cost_real + self.ema_disc_cost_fake) / 2,
                     self.inception_score - 6,
                     ]
        return np.array(state, dtype='f')

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
        all_samples = ((all_samples+1.)*255./2.).astype(np.int32)
        all_samples = all_samples.reshape((-1, 32, 32, 3))
        return inception_score.get_inception_score(list(all_samples),
                                                   splits=splits)

    def generate_images(self, step):
        feed_dict = {self.noise: self.fixed_noise_128,
                     self.is_training: False}
        samples = self.sess.run(self.fake_data, feed_dict=feed_dict)
        samples = ((samples+1.)*255./2.).astype('int32')
        task_dir = os.path.join(self.config.save_images_dir, self.exp_name)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        save_path = os.path.join(task_dir, 'images_{}.jpg'.format(step))
        save_images.save_images(samples.reshape((-1, 32, 32, 3)), save_path)


class controller_designed():
    def __init__(self, config=None):
        self.step = 0
        self.config = config

    def sample(self, state):
        self.step += 1
        action = [0, 0]
        disc = self.config.disc_iters
        gen = self.config.gen_iters
        if self.step % (disc + gen) < gen:
            action[0] =1
        else:
            action[1] = 1
        return np.array(action)

    def initialize_weights(self):
        pass

    def load_model(self, a):
        pass

    def get_weights(self):
        return [0]

    def get_gradients(self, a):
        return [0]

    def train_one_step(self, a, b):
        pass

    def save_model(self, a):
        pass
