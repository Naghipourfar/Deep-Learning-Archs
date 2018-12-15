import os

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model

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
        if layer_name == "input_1" or layer_name.endswith("_pool"):
            continue
        else:
            print(layer_name)
            layer_filters = np.array(layer_dict[layer_name].get_weights()[0])
            n_filters = layer_filters.shape[3]
            n_in_channels = layer_filters.shape[2]
            print(layer_filters.shape)
            os.makedirs("../Results/Description-5/filters/%s/" % layer_name, exist_ok=True)
            for c in range(n_filters):
                fig, axes = plt.subplots(n_in_channels, 1, figsize=(15, 15))
                plt.tick_params(labelbottom=False)
                for i in range(n_in_channels):
                    filter_window = layer_filters[:, :, i, c]
                    im = axes[i].imshow(filter_window, aspect='auto', interpolation='nearest', cmap="Blues", vmin=0,
                                        vmax=1)
                    axes[i].axis('off')
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                plt.axis('off')
                plt.savefig("../Results/Description-5/filters/%s/%s_filter_%d.pdf" % (layer_name, layer_name, c))
                plt.close()

    # -----------------------------------------------------------------
    # Part C:
    # path = "../Results/Images/"
    # image_filenames = os.listdir(path)
    # for image_filename in image_filenames:
    #     image = load_img(path + image_filename, target_size=(224, 224))
    #     image = img_to_array(image)
    #     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #     image = preprocess_input(image)
