import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard

"""
    Created by Mohsen Naghipourfar on 11/19/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def create_model():
    input_layer = Input(shape=(28, 28, 1,))
    conv_2d = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="valid", activation="relu", name="Conv_1")(
        input_layer)
    conv_2d = BatchNormalization()(conv_2d)
    pooling = MaxPooling2D(pool_size=(2, 2), padding="valid", name="MaxPooling_1")(conv_2d)
    conv_2d = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="valid", activation="relu", name="Conv_2")(
        pooling)
    conv_2d = BatchNormalization()(conv_2d)
    pooling = MaxPooling2D(pool_size=(2, 2), padding="valid", name="MaxPooling_2")(conv_2d)

    flatten = Flatten()(pooling)

    dense = Dense(256, activation="relu")(flatten)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)

    output_layer = Dense(10, activation="softmax")(dense)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(),
                  metrics=['accuracy'])
    model.summary()
    return model

def create_model_v2():
    input_layer = Input((100, ))
    dense = Dense(500, activation="linear")(input_layer)
    batch_norm = BatchNormalization()(dense)
    model = Model(input_layer, batch_norm)
    model.summary()


if __name__ == '__main__':
    model = create_model()
    model.fit()
