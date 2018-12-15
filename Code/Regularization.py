import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

"""
    Created by Mohsen Naghipourfar on 2018-12-15.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def batch_normalization_from_scratch_conv(x, n_out, training_phase, name_scope="BatchNormalization"):
    epsilon = 1e-7
    with tf.name_scope(name_scope):
        # n_out is number of filters for conv
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), trainable=True, name="Gamma")
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), trainable=True, name="Beta")

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def calculate_moments_for_conv():  # calculate mean, var for conv layer
            batch_mean = tf.reduce_mean(x, axis=[0, 1, 2], keep_dims=True)
            batch_mean = array_ops.squeeze(batch_mean, [0, 1, 2])
            m = tf.reduce_mean(x, axis=[0, 1, 2], keep_dims=True)
            squared_diffs = tf.square(x - m)
            batch_var = tf.reduce_mean(squared_diffs, axis=[0, 1, 2], keep_dims=True)
            batch_var = array_ops.squeeze(batch_var, [0, 1, 2])
            return batch_mean, batch_var

        batch_mean, batch_var = calculate_moments_for_conv()

        def mean_var_with_update_at_train_time():  # Track & save the mean and variance for test time
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training_phase,  # calculate mean, var for train/test time
                            mean_var_with_update_at_train_time,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        denominator = tf.rsqrt(var + epsilon)  # NOTE: rsqrt -> 1 / sqrt()
        normed = gamma * (x - mean) * denominator + beta
    return normed


def batch_normalization_from_scratch_dense(x, n_out, training_phase, name_scope="BatchNormalization"):
    epsilon = 1e-7
    with tf.name_scope(name_scope):
        # n_out is number of nodes in fully connected layer
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), trainable=True, name="Gamma")
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), trainable=True, name="Beta")

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def calculate_moments_for_dense():  # calculate mean, var for fully connected layer
            batch_mean = tf.reduce_mean(x, axis=[0], keep_dims=True)
            m = tf.reduce_mean(x, axis=[0], keep_dims=True)
            devs_squared = tf.square(x - m)
            batch_var = tf.reduce_mean(devs_squared, axis=[0], keep_dims=True)
            return batch_mean, batch_var

        batch_mean, batch_var = calculate_moments_for_dense()

        def mean_var_with_update_at_train_time():  # Track & save the mean and variance for test time
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training_phase,  # calculate mean, var for train/test time
                            mean_var_with_update_at_train_time,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        denominator = tf.rsqrt(var + epsilon)  # NOTE: rsqrt -> 1 / sqrt()
        normed = gamma * (x - mean) * denominator + beta
    return normed


def dropout_dense(x, keep_prob, training_phase):
    with tf.name_scope("Dropout"):
        input_shape = x.get_shape().as_list()
        m = tf.cond(training_phase,
                    lambda: np.random.binomial(1, keep_prob, size=input_shape),
                    lambda: tf.constant(keep_prob, shape=input_shape))
        tf.multiply(x, m)
