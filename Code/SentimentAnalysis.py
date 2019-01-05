import re
from random import randint

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

"""
    Created by Mohsen Naghipourfar on 2019-01-05.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def get_word2vec_vectors():
    word2vec_vectors = np.zeros((len(phrases), 300))
    for idx, word in enumerate(phrases):
        try:
            word2vec_vectors[idx] = model[word]
        except:
            continue
    return word2vec_vectors


def get_next_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        if i % 2 == 0:
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1]
    return arr, labels


def get_next_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if num <= 12499:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def get_sentiments():
    sentiments = []
    with open('../Data/stanfordSentimentTreebank/sentiment_labels.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            label = line[line.find('|') + 1:-1]
            sentiments.append(label)
    return sentiments


def load_word2vec_model(path="../Data/GoogleNews-vectors-negative300.bin"):
    return KeyedVectors.load_word2vec_format(path, binary=True)


def get_phrase_and_Ids():
    phrases = []
    phrase_IDs = []
    with open('../Data/stanfordSentimentTreebank/dictionary.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            word = line[0:line.find('|')]
            phrases.append(word)
            idx = line[line.find('|') + 1:-1]
            phrase_IDs.append(idx)
    return phrases, phrase_IDs


def get_n_words_in_sentences():
    sentences_n_words = []
    with open('../Data/stanfordSentimentTreebank/datasetSentences.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            n_words = len(line.split())
            sentences_n_words.append(n_words)
    return sentences_n_words


def clean_sentence(string):
    string = string[string.find('\t') + 1:-1]
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def sentiment_label(string):
    string = string[string.find('\t') + 1:-1]
    try:
        Index = int(phrase_IDs[phrases.index(string)])
        label = float(sentiments[Index])
    except Exception:
        return -1
    return label


def get_sentence_sentiments_and_ids():
    sent_idx = 0
    ids = np.zeros((n_sentences, max_seq_length), dtype=np.int32)
    sentence_sentiments = np.zeros(n_sentences, dtype=np.float32)

    with open('../Data/stanfordSentimentTreebank/datasetSentences.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            word_idx = 0
            label = sentiment_label(line)
            if label == -1:
                continue
            sentence_sentiments[sent_idx] = label
            cleaned_line = clean_sentence(line)
            split = cleaned_line.split()
            for word in split:
                try:
                    ids[sent_idx][word_idx] = phrases.index(word)
                except ValueError:
                    ids[sent_idx][word_idx] = 0
                word_idx += 1
                if word_idx >= max_seq_length:
                    break
            sent_idx += 1
    return sent_idx, ids


if __name__ == '__main__':
    model = load_word2vec_model()

    print("Word2Vec model has been loaded!")
    n_dim = 300
    sentences_n_words = get_n_words_in_sentences()
    n_sentences = len(sentences_n_words)
    print('The total number of sentences is', n_sentences)
    print('The total number of words in the sentences is', sum(sentences_n_words))
    print('The average number of words in the sentences is', sum(sentences_n_words) / len(sentences_n_words))

    max_seq_length = 50
    phrases, phrase_IDs = get_phrase_and_Ids()

    sentiments = get_sentiments()

    word2vec_vectors = get_word2vec_vectors()

    sent_idx, ids = get_sentence_sentiments_and_ids()

    num_available = sent_idx

    print('number of available sentences = ', num_available)

    proportion = 0.6
    train_idx = int(num_available * 0.6)

    batch_size = 64
    num_units = 256
    n_classes = 1
    n_epochs = 400

    x = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    y = tf.placeholder(tf.float32, [batch_size, n_classes])

    data = tf.Variable(tf.zeros([batch_size, max_seq_length, n_dim]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(word2vec_vectors, x)
    data = tf.cast(data, dtype=tf.float32)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(num_units)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, states = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([num_units, n_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[n_classes]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    prediction = tf.nn.softmax(prediction)

    threshold = tf.ones([batch_size, n_classes], dtype=tf.float32)

    threshold = tf.scalar_mul(0.05, threshold)

    diff = tf.abs(tf.subtract(prediction, y))

    correct_prediction = tf.less(diff, threshold)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.losses.mean_squared_error(y, prediction)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(n_epochs):
        next_batch, next_batch_labels = get_next_train_batch()

        sess.run(optimizer, feed_dict={x: next_batch, y: next_batch_labels})

        train_acc = sess.run(accuracy, feed_dict={x: next_batch, y: next_batch_labels})
        train_loss = sess.run(loss, feed_dict={x: next_batch, y: next_batch_labels})
        print("Iteration %d: Train Loss: %.4f\tTrain Accuracy: %.4f%%" % (i, train_loss, 100 * train_acc))

    iterations = 10
    for i in range(iterations):
        next_batch, next_batch_labels = get_next_test_batch()
        test_acc = sess.run(accuracy, feed_dict={x: next_batch, y: next_batch_labels})
        test_loss = sess.run(loss, feed_dict={x: next_batch, y: next_batch_labels})
        print("Iteration %d: Test Loss: %.4f\tTest Accuracy: %.4f%%" % (i, test_loss, 100 * test_acc))
