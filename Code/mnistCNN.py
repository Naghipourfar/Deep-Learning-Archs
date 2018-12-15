import os

import numpy as np
import scipy
import tensorflow as tf
from tensorflow.python.ops import array_ops

"""
    Created by Mohsen Naghipourfar on 11/22/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def max_pooling(x, pooling_shape=(2, 2), name_scope="MaxPooling_1"):
    with tf.name_scope(name_scope):
        kernel_size = [1, pooling_shape[0], pooling_shape[1], 1]
        strides = [1, pooling_shape[0], pooling_shape[1], 1]
        output = tf.nn.max_pool(x, kernel_size, strides, padding="VALID")
        return output


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


def batch_normalization(x, n_out, phase_train):
    with tf.variable_scope('BatchNormalization'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


class CNN(object):
    def __init__(self, filters=None, filter_windows=None, pooling_shapes=None, n_outputs=10,
                 save_folder=None, task_num=1):
        tf.reset_default_graph()
        if filters is None:
            filters = [64, 64]
        if filter_windows is None:
            filter_windows = [(5, 5), (5, 5)]
        if pooling_shapes is None:
            pooling_shapes = [(2, 2), (2, 2)]
        self.filters = filters
        self.filter_windows = filter_windows
        self.pooling_shapes = pooling_shapes
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="input")
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, n_outputs), name="label")
        self.keep_prob = tf.placeholder(tf.float32, name="Dropout_rate")
        self.phase_train = tf.placeholder(tf.bool, name='training_phase')
        self.convolution_weights = {}
        self.convolution_layers = {}
        self.task_num = task_num
        if task_num == 1:
            output = self.create_CNN()
        elif task_num == 2:
            output = self.create_CNN()
        elif task_num == 3:
            output = self.create_CNN()
        else:
            output = self.create_CNN_v2()
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="input")
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Conv_1")
        self.variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MaxPooling_1")
        self.variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Conv_2")
        self.variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MaxPooling_2")
        self.variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Dense_1")
        self.compile(output)
        self.save_folder = save_folder
        if os.path.exists("../Results/tensorboard/" + save_folder + "_train_" + str(task_num)):
            import shutil
            shutil.rmtree("../Results/tensorboard/" + save_folder + "_train_" + str(task_num))
            shutil.rmtree("../Results/tensorboard/" + save_folder + "_test_" + str(task_num))
        os.makedirs("../Results/tensorboard/" + save_folder + "_train_" + str(task_num), exist_ok=True)
        os.makedirs("../Results/tensorboard/" + save_folder + "_test_" + str(task_num), exist_ok=True)

    def create_CNN(self):
        max_pool = tf.reshape(self.x, shape=(-1, 28, 28, 1), name="input")
        for i in range(len(self.filters)):
            convolution = self.convolution_layer(max_pool,
                                                 filters=self.filters[i],
                                                 filter_window=self.filter_windows[i],
                                                 stride=1,
                                                 n_input_channels=max_pool.get_shape().as_list()[3],
                                                 name_scope="Conv_%d" % (i + 1))
            print(convolution.get_shape().as_list())
            convolution = batch_normalization_from_scratch_conv(convolution, self.filters[i], self.phase_train)
            # print(convolution.get_shape().as_list())
            convolution = tf.nn.relu(convolution)
            self.convolution_layers["Conv_%d" % (i + 1)] = convolution
            max_pool = max_pooling(convolution,
                                   pooling_shape=self.pooling_shapes[i],
                                   name_scope="MaxPooling_%d" % (i + 1))
        n_flatten_neurons = 1
        for i in max_pool.get_shape().as_list()[1:]:
            n_flatten_neurons *= i
        flatten = tf.reshape(max_pool, [-1, n_flatten_neurons])

        fully_connected_layer = self.dense(flatten, flatten.get_shape().as_list()[1], 256, name_scope="Dense_1")
        fully_connected_layer = batch_normalization_from_scratch_dense(fully_connected_layer, 256, self.phase_train)
        fully_connected_layer = tf.nn.relu(fully_connected_layer)
        fully_connected_layer = tf.nn.dropout(fully_connected_layer, 0.5)
        output_layer = self.dense(fully_connected_layer, 256, 10, name_scope="Output_Layer")
        return output_layer

    def create_CNN_v2(self):
        max_pool = tf.reshape(self.x, shape=(-1, 28, 28, 1), name="input")
        for i in range(len(self.filters)):
            convolution = self.convolution_layer(max_pool,
                                                 filters=self.filters[i],
                                                 filter_window=self.filter_windows[i],
                                                 stride=1,
                                                 n_input_channels=max_pool.get_shape().as_list()[3],
                                                 name_scope="Conv_%d" % (i + 1), trainable=False)
            convolution = batch_normalization_from_scratch_conv(convolution, self.filters[i], self.phase_train)
            convolution = tf.nn.relu(convolution)
            self.convolution_layers["Conv_%d" % (i + 1)] = convolution
            max_pool = max_pooling(convolution,
                                   pooling_shape=self.pooling_shapes[i],
                                   name_scope="MaxPooling_%d" % (i + 1))
        n_flatten_neurons = 1
        for i in max_pool.get_shape().as_list()[1:]:
            n_flatten_neurons *= i
        flatten = tf.reshape(max_pool, [-1, n_flatten_neurons])

        fully_connected_layer = self.dense(flatten, flatten.get_shape().as_list()[1], 256, name_scope="Dense_1", trainable=False)
        fully_connected_layer = batch_normalization_from_scratch_dense(fully_connected_layer, 256, self.phase_train)
        fully_connected_layer = tf.nn.relu(fully_connected_layer)
        fully_connected_layer = tf.nn.dropout(fully_connected_layer, 0.5)
        output_layer = self.dense(fully_connected_layer, 256, 2, name_scope="Output_Layer")
        return output_layer

    def convolution_layer(self, x, filters=64, filter_window=(5, 5), stride=1, n_input_channels=1,
                          name_scope="Conv2D_1", trainable=True):
        with tf.name_scope(name_scope):
            weights = tf.Variable(
                tf.truncated_normal(shape=[filter_window[0], filter_window[1], n_input_channels, filters], mean=0.0,
                                    stddev=0.1),
                name="Weights", trainable=trainable)
            self.convolution_weights[name_scope] = weights
            biases = tf.Variable(tf.truncated_normal(shape=[filters], mean=0.0, stddev=0.1), name="Bias",
                                 trainable=trainable)
            convolution = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding="VALID")

            convolution += biases

            output = tf.nn.relu(convolution)
            # tf.summary.image("Weights", weights)
            # tf.summary.image("Biases", biases)
            return output

    def compile(self, output):
        with tf.name_scope("Cross_Entropy"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                        logits=output))

        with tf.name_scope("Train"):
            if self.task_num == 4:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(
                    self.cross_entropy, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Output_Layer"))
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(self.cross_entropy)
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(output, axis=1),
                                          tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="train_accuracy")

        tf.summary.scalar("Cross_Entropy", self.cross_entropy)
        tf.summary.scalar("accuracy", self.accuracy)

    def dense(self, x, n_input_neurons, n_output_neurons, name_scope="Dense_1", trainable=True):
        with tf.name_scope(name_scope):
            w = tf.Variable(tf.truncated_normal(shape=(n_input_neurons, n_output_neurons),
                                                mean=0, stddev=0.1, seed=2018),
                            name="Weight", trainable=trainable)
            b = tf.Variable(tf.constant(0.1, shape=[n_output_neurons]), name="Bias", trainable=trainable)
            hidden_output = tf.add(tf.matmul(x, w), b)
            tf.summary.histogram("Weights", w)

            return hidden_output

    def fit(self, n_epochs=50000, batch_size=64, verbose=1, keep_prob=0.5):
        merge = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            train_file_writer = tf.summary.FileWriter(
                logdir="../Results/tensorboard/%s" % self.save_folder + "_train_" + str(self.task_num),
                graph=sess.graph)
            test_file_writer = tf.summary.FileWriter(
                logdir="../Results/tensorboard/%s" % self.save_folder + "_test_" + str(self.task_num),
                graph=sess.graph)

            sess.run(init)

            sess.run(tf.global_variables_initializer())
            mnist = load_data()
            test_batch_xs, test_batch_ys = mnist.validation.next_batch(10000)
            train_batch_xs, train_batch_ys = mnist.validation.next_batch(40000)
            if self.task_num == 4:
                indices = []
                for idx, test_batch_y in enumerate(test_batch_ys):
                    if test_batch_y[1] == 1 or test_batch_y[4] == 1:
                        indices.append(idx)
                test_batch_xs = test_batch_xs[indices]
                test_batch_ys = test_batch_ys[indices]
                test_batch_ys = test_batch_ys[:, [1, 4]]

                indices = []
                for idx, train_batch_y in enumerate(train_batch_ys):
                    if train_batch_y[1] == 1 or train_batch_y[4] == 1:
                        indices.append(idx)
                train_batch_xs = train_batch_xs[indices]
                train_batch_ys = train_batch_ys[indices]
                train_batch_ys = train_batch_ys[:, [1, 4]]

                for i in range(n_epochs):
                    sess.run(self.optimizer,
                             feed_dict={self.x: train_batch_xs, self.y: train_batch_ys, self.keep_prob: keep_prob,
                                        self.phase_train: True})
                    if (i + 1) % 50 == 0:
                        summaries = sess.run(merge, feed_dict={self.x: train_batch_xs, self.y: train_batch_ys,
                                                               self.keep_prob: keep_prob, self.phase_train: True})
                        train_file_writer.add_summary(summaries, i)

                        summaries = sess.run(merge, feed_dict={self.x: test_batch_xs, self.y: test_batch_ys,
                                                               self.keep_prob: keep_prob, self.phase_train: False})
                        test_file_writer.add_summary(summaries, i)

                        train_accuracy, train_loss = sess.run((self.accuracy, self.cross_entropy),
                                                              feed_dict={self.x: train_batch_xs, self.y: train_batch_ys,
                                                                         self.keep_prob: keep_prob,
                                                                         self.phase_train: True})

                        test_accuracy, test_loss = sess.run([self.accuracy, self.cross_entropy],
                                                            feed_dict={self.x: test_batch_xs, self.y: test_batch_ys,
                                                                       self.keep_prob: keep_prob,
                                                                       self.phase_train: False})
                        if verbose == 1:
                            print(
                                "Epoch: %5i\t Train Accuracy: %.4f %%\t Train Loss: %.4f\t Validation Accuracy: %.4f %%\t "
                                "Validation "
                                "Loss: %.4f" % (
                                    i + 1, 100.0 * train_accuracy, train_loss, 100.0 * test_accuracy, test_loss))
                        # save_image(sess.run(self.convolution_weights["Conv_1"]))
                return
            for i in range(n_epochs):
                train_batch_xs, train_batch_ys = mnist.train.next_batch(batch_size)
                sess.run(self.optimizer,
                         feed_dict={self.x: train_batch_xs, self.y: train_batch_ys, self.keep_prob: keep_prob,
                                    self.phase_train: True})
                if (i + 1) % 50 == 0:
                    summaries = sess.run(merge, feed_dict={self.x: train_batch_xs, self.y: train_batch_ys,
                                                           self.keep_prob: keep_prob, self.phase_train: True})
                    train_file_writer.add_summary(summaries, i)

                    summaries = sess.run(merge, feed_dict={self.x: test_batch_xs, self.y: test_batch_ys,
                                                           self.keep_prob: keep_prob, self.phase_train: False})
                    test_file_writer.add_summary(summaries, i)

                    train_accuracy, train_loss = sess.run((self.accuracy, self.cross_entropy),
                                                          feed_dict={self.x: train_batch_xs, self.y: train_batch_ys,
                                                                     self.keep_prob: keep_prob, self.phase_train: True})

                    test_accuracy, test_loss = sess.run([self.accuracy, self.cross_entropy],
                                                        feed_dict={self.x: test_batch_xs, self.y: test_batch_ys,
                                                                   self.keep_prob: keep_prob, self.phase_train: False})
                    if verbose == 1:
                        print(
                            "Epoch: %5i\t Train Accuracy: %.4f %%\t Train Loss: %.4f\t Validation Accuracy: %.4f %%\t "
                            "Validation "
                            "Loss: %.4f" % (
                                i + 1, 100.0 * train_accuracy, train_loss, 100.0 * test_accuracy, test_loss))
                    chkpt_path = "../Results/CNN_iter_%4d.ckpt" % (i + 1)
                    saver.save(sess, chkpt_path)

    def restore(self, path="../Results/CNN_iter_10000.ckpt"):
        with tf.Session() as sess:
            if self.task_num == 2:
                saver = tf.train.Saver()
                saver.restore(sess, path)
                save_image(sess.run(self.convolution_weights["Conv_1"]))
            if self.task_num == 3:
                saver = tf.train.Saver()
                saver.restore(sess, path)
                self.save_a_5_image_conv_outputs(sess)
            else:
                saver = tf.train.Saver(self.variables)
                saver.restore(sess, path)

    def save_a_5_image_conv_outputs(self, sess):
        mnist = load_data()
        test_batch_xs, test_batch_ys = mnist.validation.next_batch(10000)
        keep_prob = 0.5
        for test_batch_x, test_batch_y in zip(test_batch_xs, test_batch_ys):
            if test_batch_y[5] == 1:
                test_batch_x = np.reshape(test_batch_x, newshape=(1, 784))
                test_batch_y = np.reshape(test_batch_y, newshape=(1, 10))
                conv_1_output, conv_2_output = sess.run(
                    [self.convolution_layers["Conv_1"], self.convolution_layers["Conv_2"]],
                    feed_dict={self.x: test_batch_x, self.y: test_batch_y,
                               self.keep_prob: keep_prob,
                               self.phase_train: False})
                test_batch_x = np.reshape(test_batch_x, newshape=(28, 28))
                save_image(test_batch_x, path="../Results/", filename="5.png")
                save_image(conv_1_output, path="../Results/conv1_images/", filename="Conv_1")
                save_image(conv_2_output, path="../Results/conv2_images/", filename="Conv_2")
                break


def load_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist", one_hot=True)
    return mnist


def save_image(output, path="../Results/filter_images/", filename="filter"):
    os.makedirs(path, exist_ok=True)
    if filename is "filter":
        output = np.reshape(output, newshape=(64, 5, 5))
        for i in range(64):
            # Rescale to 0-255 and convert to uint8
            rescaled = (255.0 / output[i, :, :].max() * (output[i, :, :] - output[i, :, :].min())).astype(np.uint8)
            scipy.misc.imsave(path + "filter_" + str(i + 1) + ".png", rescaled)
    elif filename.startswith("Conv"):
        convolution = np.reshape(output, newshape=(output.shape[3], output.shape[1], output.shape[2]))
        for i in range(64):
            # Rescale to 0-255 and convert to uint8
            rescaled = (255.0 / convolution[i, :, :].max() * (
                    convolution[i, :, :] - convolution[i, :, :].min())).astype(np.uint8)
            scipy.misc.imsave(path + filename + "_" + str(i + 1) + ".png", rescaled)
    else:
        rescaled = (255.0 / output.max() * (output - output.min())).astype(np.uint8)
        scipy.misc.imsave(path + filename + ".png", rescaled)


if __name__ == '__main__':
    filters = [64, 64]
    filter_windows = [(5, 5), (5, 5)]
    pooling_shapes = [(2, 2), (2, 2)]
    model = CNN(filters=filters, filter_windows=filter_windows, pooling_shapes=pooling_shapes, n_outputs=10,
                save_folder="CNN", task_num=1)
    # model.fit(n_epochs=10000, batch_size=64, verbose=1)
    #
    # model = CNN(filters=filters, filter_windows=filter_windows, pooling_shapes=pooling_shapes, n_outputs=10,
    #             save_folder="CNN", task_num=2)
    # model.restore()
    #
    # model = CNN(filters=filters, filter_windows=filter_windows, pooling_shapes=pooling_shapes, n_outputs=10,
    #             save_folder="CNN", task_num=3)
    # model.restore()
    #
    # model = CNN(filters=filters, filter_windows=filter_windows, pooling_shapes=pooling_shapes, n_outputs=2,
    #             save_folder="CNN", task_num=3)
    # model.restore()
    # model.fit(n_epochs=10000, batch_size=64, verbose=1)
