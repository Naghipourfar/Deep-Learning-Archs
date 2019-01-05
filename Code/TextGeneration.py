import os

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import GRU, Dense, Embedding
from keras.optimizers import RMSprop

"""
    Created by Mohsen Naghipourfar on 2019-01-04.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def generate_text(model, start_string):
    num_generate = 400

    input_eval = [char_dict[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(dict_char[predicted_id])

    return start_string + ''.join(text_generated)


def load_data(path="../Data/Book.txt"):
    with open(file=path) as f:
        return f.read()


def generate_data(text, total_chars, seq_len):
    x_data = []
    y_data = []
    overlap = 5
    for i in range(0, total_chars - seq_len - overlap):  # Example of an extract of dataset: Language
        x_data.append(text[i:i + seq_len])  # Example Input Data: Language
        y_data.append(text[i+seq_len])  # Example of corresponding Target Output Data: e

    return x_data, y_data


if __name__ == '__main__':
    text = load_data()
    total_characters = len(text)
    print('Length of text: %d characters' % total_characters)

    vocab = sorted(set(text))
    n_unique_characters = len(vocab)
    print('Number of unique characters is %d' % n_unique_characters)

    char_dict = {vocab[i]: i for i in range(len(vocab))}
    dict_char = {i: vocab[i] for i in range(len(vocab))}

    seq_length = 40
    learning_rate = 0.01
    num_units = 128
    n_epochs = 20
    batch_size = 13*31

    x_data, y_data = generate_data(text, total_characters, seq_length)
    total_patterns = len(x_data)
    print("Total Patterns: ", total_patterns)

    X = np.zeros((total_patterns, n_unique_characters), dtype=np.bool)
    Y = np.zeros((total_patterns, n_unique_characters), dtype=np.bool)

    for pattern in range(total_patterns):
        for seq_pos in range(seq_length):
            vocab_index = char_dict[x_data[pattern][seq_pos]]
            X[pattern, vocab_index] = 1
        vocab_index = char_dict[y_data[pattern]]
        Y[pattern, vocab_index] = 1

    if os.path.exists("./model.h5"):
        model = tf.keras.models.load_model("model.h5")
        model.summary()

    else:
        model = Sequential()
        model.add(Embedding(n_unique_characters, 256, batch_input_shape=[batch_size, None]))
        model.add(GRU(num_units, batch_input_shape=(batch_size, None, n_unique_characters), stateful=True,
                      return_sequences=False))
        model.add(Dense(n_unique_characters, activation="softmax"))

        optimizer = RMSprop(lr=learning_rate)
        loss = "categorical_crossentropy"

        model.compile(optimizer=optimizer, loss=loss)
        model.summary()

        model.fit(X,
                  Y,
                  batch_size=batch_size,
                  epochs=n_epochs,
                  shuffle=True,
                  verbose=1)

        model.save("model.h5")
    print(generate_text(model, start_string="Once upon a time "))
