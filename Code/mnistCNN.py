import os

import tensorflow as tf

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


def convolution_layer(x, filters=64, filter_window=(5, 5), stride=1, n_input_channels=1,
                      name_scope="Conv2D_1"):
    with tf.name_scope(name_scope):
        weights = tf.Variable(
            tf.truncated_normal(shape=[filter_window[0], filter_window[1], n_input_channels, filters], mean=0.0,
                                stddev=0.2),
            name="Weights")

        biases = tf.Variable(tf.constant(0.1, shape=[filters]), name="Bias")
        convolution = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding="VALID")

        convolution += biases

        output = tf.nn.relu(convolution)
        # tf.summary.image("Weights", weights)
        # tf.summary.image("Biases", biases)
        return output


class CNN(object):
    def __init__(self, filters=None, filter_windows=None, pooling_shapes=None, n_outputs=10,
                 save_folder=None):
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
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=())
        output = self.create_CNN()
        self.compile(output)
        self.save_folder = save_folder
        if os.path.exists("../Results/tensorboard/" + save_folder + "_train"):
            import shutil
            shutil.rmtree("../Results/tensorboard/" + save_folder + "_train")
            shutil.rmtree("../Results/tensorboard/" + save_folder + "_test")
        os.mkdir("../Results/tensorboard/" + save_folder + "_train")
        os.mkdir("../Results/tensorboard/" + save_folder + "_test")

    def create_CNN(self):
        max_pool = tf.reshape(self.x, shape=(-1, 28, 28, 1), name="input")
        for i in range(len(self.filters)):
            convolution = convolution_layer(max_pool,
                                            filters=self.filters[i],
                                            filter_window=self.filter_windows[i],
                                            stride=1,
                                            n_input_channels=max_pool.get_shape().as_list()[3],
                                            name_scope="Conv_%d" % (i + 1))
            max_pool = max_pooling(convolution,
                                   pooling_shape=self.pooling_shapes[i],
                                   name_scope="MaxPooling_%d" % (i + 1))
        n_flatten_neurons = 1
        for i in max_pool.get_shape().as_list()[1:]:
            n_flatten_neurons *= i
        flatten = tf.reshape(max_pool, [-1, n_flatten_neurons])

        fully_connected_layer = self.dense(flatten, flatten.get_shape().as_list()[1], 256, activation=tf.nn.relu,
                                           name_scope="Dense_1")
        fully_connected_layer = tf.nn.dropout(fully_connected_layer, 0.5)
        output_layer = self.dense(fully_connected_layer, 256, 10, activation=tf.nn.softmax_cross_entropy_with_logits_v2,
                                  name_scope="Output_Layer")
        return output_layer

    def compile(self, output):
        with tf.name_scope("Cross_Entropy"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                           logits=output))

        with tf.name_scope("Train"):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(
                self.cross_entropy)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(output, axis=1),
                                          tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="train_accuracy")

        tf.summary.scalar("Cross_Entropy", self.cross_entropy)
        tf.summary.scalar("accuracy", self.accuracy)

    def dense(self, x, n_input_neurons, n_output_neurons, activation=tf.nn.relu, name_scope="Dense_1"):
        with tf.name_scope(name_scope):
            w = tf.Variable(tf.truncated_normal(shape=(n_input_neurons, n_output_neurons),
                                                mean=0, stddev=0.2, seed=1),
                            name="Weight")
            b = tf.Variable(tf.constant(0.1, shape=[n_output_neurons]), name="Bias")
            hidden_output = tf.add(tf.matmul(x, w), b)
            if activation == tf.nn.softmax_cross_entropy_with_logits_v2:
                activation_output = tf.nn.softmax(logits=hidden_output)
            else:
                activation_output = activation(hidden_output)
            tf.summary.histogram("Weights", w)

            return activation_output

    def fit(self, n_epochs=50000, batch_size=64, verbose=1, keep_prob=0.5):
        merge = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            train_file_writer = tf.summary.FileWriter(logdir="../Results/tensorboard/%s" % self.save_folder + "_train",
                                                      graph=sess.graph)
            test_file_writer = tf.summary.FileWriter(logdir="../Results/tensorboard/%s" % self.save_folder + "_test",
                                                     graph=sess.graph)

            sess.run(init)

            sess.run(tf.global_variables_initializer())
            mnist = load_data()
            test_batch_xs, test_batch_ys = mnist.validation.next_batch(10000)
            for i in range(n_epochs):
                train_batch_xs, train_batch_ys = mnist.train.next_batch(batch_size)
                sess.run(self.optimizer,
                         feed_dict={self.x: train_batch_xs, self.y: train_batch_ys, self.keep_prob: keep_prob})
                if (i + 1) % 1 == 0:
                    summaries = sess.run(merge, feed_dict={self.x: train_batch_xs, self.y: train_batch_ys})
                    train_file_writer.add_summary(summaries, i)

                    summaries = sess.run(merge, feed_dict={self.x: test_batch_xs, self.y: test_batch_ys})
                    test_file_writer.add_summary(summaries, i)

                    train_accuracy, train_loss = sess.run((self.accuracy, self.cross_entropy),
                                                          feed_dict={self.x: train_batch_xs, self.y: train_batch_ys})

                    test_accuracy, test_loss = sess.run([self.accuracy, self.cross_entropy],
                                                        feed_dict={self.x: test_batch_xs, self.y: test_batch_ys})
                    if verbose == 1:
                        print(
                            "Epoch: %5i\t Train Accuracy: %.4f %%\t Train Loss: %.4f\t Validation Accuracy: %.4f %%\t "
                            "Validation "
                            "Loss: %.4f" % (
                                i + 1, 100.0 * train_accuracy, train_loss, 100.0 * test_accuracy, test_loss))


def load_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist", one_hot=True)
    return mnist


if __name__ == '__main__':
    filters = [64, 64]
    filter_windows = [(5, 5), (5, 5)]
    pooling_shapes = [(2, 2), (2, 2)]
    model = CNN(filters=filters, filter_windows=filter_windows, pooling_shapes=pooling_shapes, n_outputs=10,
                save_folder="CNN")
    model.fit(n_epochs=50000, batch_size=64, verbose=1)
