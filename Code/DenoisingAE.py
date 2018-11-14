import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

"""
    Created by Mohsen Naghipourfar on 3/26/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""
"""
    This is a network which is learning the mnist noised data!
"""

# Hyper-Parameters
neurons = {
    'input': 784,
    'encode': 16,
    'decode': 784
}
n_epochs = 1
batch_size = 256
learning_rate = 0.001
noise_factor = 0.5

# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape Data
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Normalize Data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Add noise to the data
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Make our Model
input_image = Input(shape=(28, 28, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

# Add Sparsity to Model (!)
# sparse_encoded = Dense(32, activity_regularizer=l1(10e-5))(encoded)


x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Build the model
auto_encoder = Model(input_image, decoded)

auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

auto_encoder.summary()

# Train The Model
auto_encoder.fit(x=x_train_noisy,
                 y=x_train,
                 epochs=n_epochs,
                 batch_size=batch_size,
                 shuffle=True,
                 validation_data=(x_test, x_test))

# Predict The test data
decoded_images = auto_encoder.predict(x_test_noisy)

# Visualization some results
number_of_digits_to_show = 30
plt.figure(figsize=(60, 4))

for i in range(number_of_digits_to_show):
    # display original
    ax = plt.subplot(2, number_of_digits_to_show, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
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
