import os

import tensorflow as tf

"""
    Created by Mohsen Naghipourfar on 11/11/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


class Model(object):
    def __init__(self, layers, learning_rate, hidden_activation=tf.sigmoid,
                 optimization_algorithm=tf.train.GradientDescentOptimizer, save_folder="Graph"):
        tf.reset_default_graph()
        self.layers = layers
        self.layer_outputs = {}
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, layers[0]), name="input")
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, layers[-1]), name="label")
        self.optimizer = None
        self.cross_entropy = None
        self.accuracy = None
        self.optimization_algorithm = optimization_algorithm
        self.save_folder = save_folder
        os.mkdir("../Results/tensorboard/" + save_folder + "_train")
        os.mkdir("../Results/tensorboard/" + save_folder + "_test")
        self.MLP()

    def MLP(self):
        for idx, n_neuron in enumerate(self.layers[:-1]):
            if idx == 0:
                continue
            self.dense(name_scope="Hidden_Layer_%d" % idx, idx=idx, n_neuron=n_neuron,
                       activation=self.hidden_activation)
        self.dense(name_scope="Output_Layer", idx=len(self.layers) - 1, n_neuron=self.layers[-1],
                   activation=tf.nn.softmax_cross_entropy_with_logits_v2)

        self.compile()

    def compile(self):
        with tf.name_scope("Cross_Entropy"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                           logits=self.layer_outputs['output']))

        with tf.name_scope("Train"):
            self.optimizer = self.optimization_algorithm(learning_rate=self.learning_rate).minimize(
                self.cross_entropy)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.layer_outputs['output'], axis=1),
                                          tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="train_accuracy")

        tf.summary.scalar("Cross_Entropy", self.cross_entropy)
        tf.summary.scalar("accuracy", self.accuracy)

    def dense(self, name_scope, idx, n_neuron, activation=tf.sigmoid):
        with tf.name_scope(name_scope):
            w = tf.Variable(tf.truncated_normal(shape=(self.layers[idx - 1], n_neuron),
                                                mean=0, stddev=1.5, seed=1),
                            name="Weight")
            b = tf.Variable(tf.zeros([n_neuron]), name="Bias")
            if idx == 1:
                hidden_output = tf.add(tf.matmul(self.x, w), b)
            else:
                hidden_output = tf.add(tf.matmul(self.layer_outputs[idx - 1], w), b)
            if activation == tf.nn.softmax_cross_entropy_with_logits_v2:
                activation_output = tf.nn.softmax(logits=hidden_output)
            else:
                activation_output = activation(hidden_output)
            tf.summary.histogram("Weights", w)
        if idx == len(self.layers) - 1:
            self.layer_outputs['output'] = activation_output
        else:
            self.layer_outputs[idx] = activation_output

    def fit(self, n_epochs=50000, batch_size=64, verbose=1):
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
                sess.run(self.optimizer, feed_dict={self.x: train_batch_xs, self.y: train_batch_ys})
                if (i + 1) % 1000 == 0:
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
    layers = [[784, 50, 10],
              [784, 500, 10],
              [784, 50, 50, 10],
              ]

    activations = [tf.sigmoid, tf.tanh]
    learning_rates = [0.5, 1.0, 0.01, 0.001]
    batch_sizes = [16, 64, 128, 256]
    optimization_algorithms = [tf.train.GradientDescentOptimizer, tf.train.RMSPropOptimizer, tf.train.AdamOptimizer]

    # model = Model(layers=layers[0], learning_rate=learning_rates[0], hidden_activation=tf.sigmoid, save_folder="Q1")
    # model.fit(n_epochs=50000, batch_size=batch_sizes[1], verbose=1)
    #
    # model = Model(layers=layers[0], learning_rate=learning_rates[0], hidden_activation=tf.tanh, save_folder="Q2")
    # model.fit(n_epochs=50000, batch_size=batch_sizes[1], verbose=1)

    model = Model(layers=layers[1], learning_rate=learning_rates[0], hidden_activation=tf.sigmoid, save_folder="Q3")
    model.fit(n_epochs=50000, batch_size=batch_sizes[1], verbose=1)

    # model = Model(layers=layers[2], learning_rate=learning_rates[0], hidden_activation=tf.tanh, save_folder="Q4")
    # model.fit(n_epochs=50000, batch_size=batch_sizes[1], verbose=1)

    # for learning_rate in learning_rates:
    #     model = Model(layers=layers[2], learning_rate=learning_rate, hidden_activation=tf.tanh,
    #                   save_folder="Q6-lr=%.3f" % learning_rate)
    #     model.fit(n_epochs=50000, batch_size=batch_sizes[1], verbose=1)
    #
    # for batch_size in batch_sizes:
    #     model = Model(layers=layers[2], learning_rate=learning_rates[0], hidden_activation=tf.tanh,
    #                   save_folder="Q7-batch_size=%d" % batch_size)
    #     model.fit(n_epochs=50000, batch_size=batch_size, verbose=1)
    # for optimization_algorithm in optimization_algorithms:
    #     model = Model(layers=layers[2], learning_rate=learning_rates[0], hidden_activation=tf.tanh,
    #                   save_folder="Q8-optAlg=%s" % str(optimization_algorithm.__name__))
    #     model.fit(n_epochs=50000, batch_size=batch_sizes[1], verbose=1)
