import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.contrib import rnn

"""
    Created by Mohsen Naghipourfar on 2019-01-03.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""
n_characters = 4  # a, b, N, e

char_dict = {"a": 0, "b": 1, "N": 2, "e": 3}
dict_char = {0: "a", 1: "b", 2: "N", 3: "e"}

# Outputs and states of LSTM
outputs = None
states = None


def convert_seq(seq):
    converted_seq = [0 for _ in range(len(seq))]
    for i, ch in enumerate(seq):
        converted_seq[i] = char_dict[ch]
    return converted_seq


def generate_pattern(k=10, pattern_type=1):
    if pattern_type == 1:
        pattern = ("a" * k) + "N" + ("b" * k)
    else:
        pattern = ("ab" * k) + "N" + ("ba" * k)
    return pattern


def generate_test_pattern(k=10, pattern_type=1):
    if pattern_type == 1:
        pattern = ("a" * k) + "N"
    else:
        pattern = ("ab" * k) + "N"
    return pattern


def generate_data(max_k=11, pattern_type=1):
    x_data, y_data = [], []
    for i in range(1, max_k + 1):
        input_seq = generate_pattern(i, pattern_type=pattern_type)
        output_seq = input_seq[1:] + "e"
        x_data.append(convert_seq(input_seq))
        y_data.append(convert_seq(output_seq))
    return x_data, y_data


def preprocess_sample(x, y):
    x = np.array(x, dtype=np.float32)
    x /= float(n_characters)
    x = np.reshape(x, (1, len(x), 1))
    y = np_utils.to_categorical(y, num_classes=n_characters)
    return x, y


def generate_cell_state_image(cell_state, filename="cell_state(k=15).pdf"):
    plt.figure(1, figsize=(20, 20))
    plt.imshow(cell_state, interpolation="nearest", cmap="Blues")
    plt.savefig('../Results/HW6/' + filename)
    plt.close()


class RNN(object):
    def __init__(self, num_units):
        tf.reset_default_graph()
        self.num_units = num_units
        self.x = tf.placeholder(tf.float32, [1, None, 1])
        self.y = tf.placeholder(tf.float32, [None, n_characters])
        self.weights = {
            "out": tf.Variable(tf.truncated_normal(mean=0.0, stddev=0.01, shape=(self.num_units, n_characters))),
        }

        self.biases = {
            "out": tf.Variable(tf.truncated_normal(mean=0.0, stddev=0.01, shape=[n_characters]))
        }
        self.sess = tf.InteractiveSession()
        self.compile(learning_rate=0.001)

    def compile(self, learning_rate):
        def LSTM(x, num_units=10):
            global outputs, states
            lstm_cell = rnn.BasicLSTMCell(num_units=num_units)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

            return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

        self.model = LSTM(self.x, num_units=self.num_units)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def fit(self, x_data, y_data, n_epochs, verbose=1):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(n_epochs + 1):
            batch_x, batch_y = None, None
            for j in range(10):
                batch_x, batch_y = x_data[j], y_data[j]
                batch_x, batch_y = preprocess_sample(batch_x, batch_y)

                self.sess.run([self.optimizer], feed_dict={self.x: batch_x, self.y: batch_y})
            if i % 10 == 0 and verbose == 1:
                train_acc, train_loss = self.sess.run([self.accuracy, self.loss],
                                                      feed_dict={self.x: batch_x, self.y: batch_y})
                print("Iteration %d: Train Loss: %.4f\tTrain Accuracy: %.4f%%" % (i, train_loss, 100 * train_acc))

    def evaluate(self, pattern_type=1):
        # Test model for K >= 11
        if pattern_type == 1:
            for k in range(11, 21):
                x_data = generate_test_pattern(k, pattern_type=pattern_type)
                y_data = generate_pattern(k, pattern_type=pattern_type)
                X = x_data
                x_data = convert_seq(x_data)
                y_data = convert_seq(y_data)
                print("*" * 100)
                batch_x, batch_y = preprocess_sample(x_data, y_data)
                pred_seq = self.sess.run([self.model], feed_dict={self.x: batch_x})
                pred_seq = np.argmax(pred_seq[0], axis=1)
                pred_seq = [dict_char[idx] for idx in pred_seq]
                pred_seq = "".join(pred_seq)
                pred_seq = X + pred_seq
                # sample_loss, sample_acc = self.sess.run([self.loss, self.accuracy],
                #                                         feed_dict={self.x: batch_x, self.y: batch_y})
                true_seq = generate_pattern(k, pattern_type=pattern_type)
                print("Test for K = %d: \tPrediction:%s\tTrueSeq:%s" % (
                    k, pred_seq, true_seq))
        if pattern_type == 2:
            for k in range(16, 21):
                x_data = generate_test_pattern(k, pattern_type=pattern_type)
                y_data = generate_pattern(k, pattern_type=pattern_type)
                X = x_data
                x_data = convert_seq(x_data)
                y_data = convert_seq(y_data)
                print("*" * 100)
                batch_x, batch_y = preprocess_sample(x_data, y_data)
                pred_seq = self.sess.run([self.model], feed_dict={self.x: batch_x})
                pred_seq = np.argmax(pred_seq[0], axis=1)
                pred_seq = [dict_char[idx] for idx in pred_seq]
                pred_seq = "".join(pred_seq)
                pred_seq = X + pred_seq
                # sample_loss, sample_acc = self.sess.run([self.loss, self.accuracy],
                #                                         feed_dict={self.x: batch_x, self.y: batch_y})
                true_seq = generate_pattern(k, pattern_type=pattern_type)
                print("Test for K = %d: \tPrediction:%s\tTrueSeq:%s" % (
                    k, pred_seq, true_seq))

    def get_cell_state(self, x_data, y_data):
        batch_x, batch_y = x_data, y_data
        batch_x, batch_y = preprocess_sample(batch_x, batch_y)
        sample_loss, sample_acc = self.sess.run([self.loss, self.accuracy],
                                                feed_dict={self.x: batch_x, self.y: batch_y})
        sample_cell_state, sample_outputs = self.sess.run([states, outputs],
                                                          feed_dict={self.x: batch_x, self.y: batch_y})
        sample_outputs = np.array(sample_outputs)
        generate_cell_state_image(sample_outputs[0, :, :],
                                  filename="cell_state(k=15)(num_units=%d).pdf" % self.num_units)


if __name__ == '__main__':
    # Loading Data
    x_data, y_data = generate_data(max_k=20)
    n_patterns = len(x_data)
    print("Number of all patterns are %d" % n_patterns)

    # RNN (LSTM) with 10 hidden units
    rnn_model = RNN(num_units=10)
    rnn_model.fit(x_data, y_data, n_epochs=1000)  # Train Model
    rnn_model.evaluate(pattern_type=1)  # Test for K >= 11 patterns

    # Generate Cell State Diagram for k = 15
    k = 14
    rnn_model.get_cell_state(x_data[k], y_data[k])

    # Loading Data
    x_data, y_data = generate_data(max_k=15, pattern_type=2)
    n_patterns = len(x_data)
    print("Number of all patterns are %d" % n_patterns)
    # RNN (LSTM) with 20 hidden units
    rnn_model = RNN(num_units=20)
    rnn_model.fit(x_data, y_data, n_epochs=1000)  # Train Model
    rnn_model.evaluate(pattern_type=2)  # Test for K >= 11 patterns

    # Generate Cell State Diagram for k = 15
    k = 14
    rnn_model.get_cell_state(x_data[k], y_data[k])
