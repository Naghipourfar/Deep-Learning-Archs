import matplotlib.pyplot as plt
import numpy as np

"""
    Created by Mohsen Naghipourfar on 10/10/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def f(x):
    import math
    return math.sqrt(x ** 2 + .001)


def huber(x, c=1):
    if abs(x) < c:
        return 0.5 * (x ** 2)
    else:
        return c * abs(x) - ((c ** 2) / 2)


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def tanh(x):
    import math
    return math.tanh(x)


if __name__ == '__main__':
    n_points = 250
    xs = np.arange(start=-10, stop=10, step=0.01)
    ys = []
    zs = []
    for x in xs:
        ys.append(huber(x))
        zs.append(sign(x))
    plt.close("all")
    plt.figure(figsize=(15, 10))
    plt.plot(xs, ys, '-', label="Huber Function")
    plt.plot(xs, abs(xs), '-', label="Absolute Function")
    plt.grid()
    plt.legend()
    plt.savefig("../Results/Huber.pdf")
    plt.show()
