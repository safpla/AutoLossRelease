# __author__ == 'Haowen Xu'
# __data__ == '04-30-2018'

import numpy as np
import tensorflow as tf

stdev = 0.02
def _uniform(stdev, size):
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size).astype('float32')

def _normal(stdev, size):
    return np.random.normal(scale=stdev, size=size).astype('float32')

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.))

def linear(input,
           output_size,
           name='LN',
           initialization='normal',
           gain=1.,
           stdev=0.02):
    input_shape = input.get_shape().as_list()
    input_size = input_shape[1]

    with tf.variable_scope(name) as scope:
        if initialization == 'glorot' or initialization == None:
            weight_values = _uniform(np.sqrt(2./(input_size + output_size)),
                                     (input_size, output_size)
                                     )
        elif initialization == 'normal':
            weight_values = _normal(stdev, (input_size, output_size))
        else:
            raise Exception('Invalid initialization!')
        weight_values *= gain
        w = tf.get_variable('W', initializer=weight_values, dtype=tf.float32)
        b = _bias_variable('b', [output_size])
        return tf.matmul(input, w) + b

def batchnorm(input, is_training=True, name='BN'):
    epsilon = 1e-5
    momentum = 0.95
    output = tf.contrib.layers.batch_norm(
        input,
        decay=momentum,
        updates_collections=None,
        epsilon=epsilon,
        is_training=is_training,
        scope=name
    )
    return output

def deconv2d(input, output_shape, k_h=5, k_w=5, s_h=2, s_w=2, name='deconv',
             initialization='normal'):
    # ----Filter shape: [height, width, output_channels, input_channels]----
    # ----Output shape: [batch_size, height, width, channels]----
    input_shape = input.get_shape().as_list()
    input_shape_t = tf.shape(input)
    fan_in = input_shape[-1] * k_h * k_w / (s_h * s_w)
    fan_out = output_shape[-1] * k_h * k_w
    output_shape[0] = input_shape_t[0]

    with tf.variable_scope(name):
        if initialization == 'he':
            weight_values = _uniform(np.sqrt(4./(fan_in + fan_out)),
                (k_h, k_w, output_shape[-1], input_shape[-1]))
        elif initialization == 'normal':
            weight_values = _normal(stdev,
                (k_h, k_w, output_shape[-1], input_shape[-1]))
        else:
            raise Exception('Invalid initialization!')
        w = tf.get_variable('W', initializer=weight_values, dtype=tf.float32)
        b = _bias_variable('b', [output_shape[-1]])
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape,
                                        strides=[1, s_h, s_w, 1])
        deconv = tf.nn.bias_add(deconv, b)
        return deconv

def conv2d(input, output_dim, k_h=5, k_w=5, s_h=2, s_w=2, name='conv',
           initialization='normal'):
    # ----Filter shape: [height, width, input_channels, output_channels]----
    # ----Output shape: [batch_size, classes]----
    input_shape = input.get_shape().as_list()
    fan_in = input_shape[-1] * k_h * k_w
    fan_out = output_dim * k_h * k_w / (s_h * s_w)

    with tf.variable_scope(name):
        if initialization == 'he':
            weight_values = _uniform(np.sqrt(4./(fan_in + fan_out)),
                (k_h, k_w, input_shape[-1], output_dim))
        elif initialization == 'normal':
            weight_values = _normal(stdev,
                (k_h, k_w, input_shape[-1], output_dim))
        else:
            raise Exception('Invalid initialization!')
        w = tf.get_variable('W', initializer=weight_values, dtype=tf.float32)
        b = _bias_variable('b', [output_dim])
        conv = tf.nn.conv2d(input, w, strides=[1, s_h, s_w, 1], padding='SAME')
        return tf.nn.bias_add(conv, b)




if __name__ == '__main__':
    input = tf.constant([[2,2,2,2],[1,1,1,1]], dtype=tf.float32)
    output_size = 2
    output = batchnorm(input, output_size)
    tvars = tf.trainable_variables()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r = sess.run([input, output])
        w = sess.run(tvars)
        print(r)
        print(w)

