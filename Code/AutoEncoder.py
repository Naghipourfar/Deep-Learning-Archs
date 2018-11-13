import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model

"""
    Created by Mohsen Naghipourfar on 3/26/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""

"""
    Building an AutoEncoder 
"""

# Hyper-Parameters
neurons = {
    'input': 784,
    'encode': 16,
    'decode': 784
}
n_epochs = 10
batch_size = 256
learning_rate = 0.001

# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape Data
x_train = x_train.reshape((len(x_train), np.product(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.product(x_test.shape[1:])))

# Normalize Data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Make our Model
input_image = Input(shape=(neurons['input'],))

encode_layer = Dense(neurons['encode'], activation='relu')(input_image)

decode_layer = Dense(neurons['decode'], activation='sigmoid')(encode_layer)

auto_encoder = Model(input_image, decode_layer)

auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
auto_encoder.summary()

# Train The Model
auto_encoder.fit(x=x_train,
                 y=x_train,
                 batch_size=batch_size,
                 epochs=n_epochs,
                 shuffle=True,
                 validation_data=(x_test, x_test))

# Predict with model
decoded_images = auto_encoder.predict(x_test)

# Visualization some results
number_of_digits_to_show = 10
plt.figure(figsize=(20, 4))

for i in range(number_of_digits_to_show):
    # display original
    ax = plt.subplot(2, number_of_digits_to_show, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, number_of_digits_to_show, i + 1 + number_of_digits_to_show)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
