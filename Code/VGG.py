import os
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils.vis_utils import plot_model
from keras_preprocessing.image import load_img, img_to_array

"""
    Created by Mohsen Naghipourfar on 2018-12-15.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

if __name__ == '__main__':
    # -----------------------------------------------------------------
    # Part A:
    model = VGG16()
    model.summary()
    plot_model(model, "../Results/Description-5/VGG16.pdf")

    # -----------------------------------------------------------------
    # Part B:
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    plt.close("all")
    for layer_name, layer in layer_dict.items():
        if layer_name == "input_1" or layer_name.endswith("_pool") or layer_name.startswith(
                "f") or layer_name.startswith("predictions"):
            continue
        else:
            print(layer_name)
            layer_filters = np.array(layer_dict[layer_name].get_weights()[0])
            n_filters = layer_filters.shape[3]
            n_in_channels = layer_filters.shape[2]
            print(layer_filters.shape)
            os.makedirs("../Results/Description-5/filters/%s/" % layer_name, exist_ok=True)
            for c in range(n_filters):
                filter_window = np.reshape(layer_filters[:, :, :, c], newshape=(-1, 9))
                plt.figure(figsize=(10, 10))
                plt.axis("off")
                im = plt.imshow(filter_window, aspect="auto", interpolation="nearest", cmap="Blues")
                cbar_ax = plt.axes([.9, 0.15, 0.05, 0.7])
                cbar = plt.colorbar(im, cax=cbar_ax)
                plt.savefig("../Results/Description-5/filters/%s/filter_%d.pdf" % (layer_name, c))
                plt.close()

    # -----------------------------------------------------------------
    # Part C:
    # layer_name = "encoded"
    # encoded_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    # encoded_output = encoded_layer_model.predict(df_m_rna)
    path = "../Results/Images/"
    layer_names = ["block1_pool", "block4_conv3"]
    image_filenames = os.listdir(path)
    for image_filename in image_filenames:
        image = load_img(path + image_filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        for layer_name in layer_names:
            new_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            filter_windows = new_model.predict(image)[0, :, :, :]
            for i in range(filter_windows.shape[2]):
                filter_window = filter_windows[:, :, i]
                plt.figure(figsize=(10, 10))
                plt.axis("off")
                im = plt.imshow(filter_window, aspect="auto", interpolation="nearest", cmap="Blues")
                cbar_ax = plt.axes([.9, 0.15, 0.05, 0.7])
                cbar = plt.colorbar(im, cax=cbar_ax)
                os.makedirs("../Results/Description-5/images_filters/%s/" % (image_filename.split(".")[0]),
                            exist_ok=True)
                plt.savefig(
                    "../Results/Description-5/images_filters/%s/%s_%d.pdf" % (
                    image_filename.split(".")[0], layer_name, i))
                plt.close()
