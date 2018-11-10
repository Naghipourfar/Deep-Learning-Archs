from keras.layers import Input, Conv2D
from keras.models import Model

"""
    Created by Mohsen Naghipourfar on 11/19/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def create_model():
    input_layer = Input((65, 65, 3))
    conv_1 = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=(65, 65, 3))(input_layer)
    model = Model(input_layer, conv_1)
    model.summary()


if __name__ == '__main__':
    create_model()
