import math

import matplotlib.pyplot as plt
import numpy as np

"""
    Created by Mohsen Naghipourfar on 11/2/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def rastrigin(x, y):
    return 20 + (x ** 2) + (y ** 2) - (10 * math.cos(2 * math.pi * x)) - (10 * math.cos(2 * math.pi * y))


def rastrigin_gradient_x(x, y):
    return (2 * x) + (20 * math.pi * math.sin(2 * math.pi * x))


def rastrigin_gradient_y(x, y):
    return (2 * y) + (20 * math.pi * math.sin(2 * math.pi * y))


def rastrigin_hessian(x, y):
    return np.array([[2 + 40 * (math.pi ** 2) * math.cos(2 * math.pi * x), 0],
                     [0, 2 + 40 * (math.pi ** 2) * math.cos(2 * math.pi * y)]])


def ackley(x, y):
    return -20 * math.exp(-0.2 * math.sqrt(0.5 * ((x ** 2) + (y ** 2)))) - math.exp(
        math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)) + math.exp(1) + 20


def ackley_gradient_x(x, y):
    return (2 * x / math.sqrt(0.5 * ((x ** 2) + (y ** 2)))) * math.exp(-0.2 * math.sqrt(0.5 * ((x ** 2) + (y ** 2)))) + \
           2 * math.pi * math.sin(2 * math.pi * x) * math.exp(math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))


def ackley_gradient_y(x, y):
    return (2 * y / math.sqrt(0.5 * ((x ** 2) + (y ** 2)))) * math.exp(-0.2 * math.sqrt(0.5 * ((x ** 2) + (y ** 2)))) + \
           2 * math.pi * math.sin(2 * math.pi * y) * math.exp(math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))


def d2ackley_dx2(x, y):
    expr_1 = math.sqrt(0.5 * ((x ** 2) + (y ** 2)))
    expr_2 = math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)
    A = math.exp(-0.2 * expr_1)
    B = math.exp(expr_2)
    return -20 * A * ((-0.1 * x / expr_1) ** 2) + 2 * A * (expr_1 - ((x ** 2) / (2 * expr_1))) / (expr_1 ** 2) + 4 * (
            math.pi ** 2) * math.cos(2 * math.pi * x) * B - 4 * (math.pi ** 2) * (
                   math.sin(2 * math.pi * x) ** 2) * B


def d2ackley_dxdy(x, y):
    expr_1 = math.sqrt(0.5 * ((x ** 2) + (y ** 2)))
    expr_2 = math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)
    A = math.exp(-0.2 * expr_1)
    B = math.exp(expr_2)
    return -20 * A * (-0.1 * x / expr_1) * (-0.1 * y / expr_1) - 20 * A * (-0.1 * x * y / (2 * (expr_1 ** 3))) - 4 * (
            math.pi ** 2) * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y) * B


def d2ackley_dydx(x, y):
    return d2ackley_dxdy(x, y)


def d2ackley_dy2(x, y):
    return d2ackley_dx2(y, x)


def ackley_hessian(x, y):
    return np.array([[d2ackley_dx2(x, y), d2ackley_dxdy(x, y)],
                     [d2ackley_dydx(x, y), d2ackley_dy2(x, y)]])


def dA_dx(x, y):
    expr = math.sqrt(0.5 * ((x ** 2) + (y ** 2)))
    return math.exp(-0.2 * expr) * -0.2 * (x / (2 * expr))


def dA_dy(x, y):
    expr = math.sqrt(0.5 * ((x ** 2) + (y ** 2)))
    return math.exp(-0.2 * expr) * -0.2 * (y / (2 * expr))


def dB_dx(x, y):
    B = math.exp(math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))
    return -2 * math.pi * math.sin(2 * math.pi * x) * B


def dB_dy(x, y):
    B = math.exp(math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))
    return -2 * math.pi * math.sin(2 * math.pi * y) * B


def levi(x, y):
    return (math.sin(3 * math.pi * x) ** 2) + (((x - 1) ** 2) * (1 + (math.sin(3 * math.pi * y) ** 2))) + (
            ((y - 1) ** 2) * (1 + (math.sin(2 * math.pi * y) ** 2)))


def levi_gradient_x(x, y):
    return 3 * math.pi * math.sin(6 * math.pi * x) + (2 * (x - 1) * (1 + (math.sin(3 * math.pi * y) ** 2)))


def levi_gradient_y(x, y):
    return (3 * math.pi * ((x - 1) ** 2) * math.sin(6 * math.pi * y)) + \
           (2 * (y - 1) * (1 + (math.sin(3 * math.pi * y) ** 2))) + \
           (((y - 1) ** 2) * 2 * math.pi * math.sin(4 * math.pi * y))


def levi_hessian(x, y):
    pi = math.pi
    return np.array([[18 * (pi ** 2) * math.cos(6 * pi * x) + 2 * (1 + math.sin(3 * pi * y) ** 2),
                      2 * (x - 1) * (3 * pi * math.sin(6 * pi * y))],
                     [2 * (x - 1) * (3 * pi * math.sin(6 * pi * y)),
                      18 * (pi ** 2) * ((x - 1) ** 2) * math.cos(6 * pi * y) + 2 * (
                              1 + math.sin(2 * pi * y) ** 2) + 2 * (y - 1) * (2 * pi * math.sin(4 * pi * y)) + 2 * (
                              y - 1) * (2 * pi * math.sin(4 * pi * y)) + ((y - 1) ** 2) * (
                              8 * (pi ** 2) * math.cos(4 * pi * y))]])


def bukin(x, y):
    return 100 * math.sqrt(abs(y - (0.01 * (x ** 2)))) + 0.01 * abs(x + 10)


def bukin_gradient_x(x, y):
    return (sign(0.01 * x ** 2 - y) * x / math.sqrt(abs(y - 0.01 * x ** 2))) + 0.01 * sign(x + 10)


def bukin_gradient_y(x, y):
    return 50 * sign(y - 0.01 * x ** 2) / math.sqrt(abs(y - 0.01 * x ** 2))


def bukin_hessian(x, y):
    expr = y - 0.01 * (x ** 2)
    return np.array([[-1 * (math.sqrt(abs(expr)) + ((x ** 2) * sign(expr) / math.sqrt(abs(expr)))) * sign(expr) / abs(
        expr), -1 * x * sign(expr) * (-1 * sign(expr) / (2 * math.sqrt(abs(expr) ** 3)))],
                     [50 * sign(expr) * (0.01 * x * sign(expr) / math.sqrt(abs(expr) ** 3)),
                      50 * sign(expr) * (-1 * sign(expr) / (2 * math.sqrt(abs(expr) ** 3)))]], dtype=np.float64)


def n_d_rastrigin(X):
    return 20 + sum([x ** 2 - 10 * math.cos(2 * math.pi * x) for x in X])


def n_d_rastrigin_gradient(X, idx):
    return 2 * X[idx] + 20 * math.pi * math.sin(2 * math.pi * X[idx])


def n_d_rastrigin_hessian(X):
    H = np.zeros(shape=(len(X), len(X)))
    for i in range(len(X)):
        H[i, i] = 2 + 40 * (math.pi ** 2) * math.cos(2 * math.pi * X[i])
    return H


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class GradientDescent(object):
    def __init__(self, loss, loss_gradients, learning_rate=0.01):
        self.loss = loss
        self.loss_gradients = loss_gradients
        self.learning_rate = learning_rate
        self.n_variables = len(loss_gradients)

    def get_next_point(self, start_point, n_epochs=250):
        new_point = [0 for _ in range(self.n_variables)]
        for i in range(n_epochs):
            for j in range(self.n_variables):
                if self.loss == n_d_rastrigin:
                    new_point[j] = start_point[j] - self.learning_rate * self.loss_gradients[j](start_point, j)
                else:
                    new_point[j] = start_point[j] - self.learning_rate * self.loss_gradients[j](*start_point)
        if self.loss == n_d_rastrigin:
            return new_point, self.loss(new_point)
        return new_point, self.loss(*new_point)


class Nesterov(GradientDescent):
    def __init__(self, loss, loss_gradients, learning_rate=0.01, momentum=0.8):
        super().__init__(loss, loss_gradients, learning_rate)
        self.momentum = momentum
        self.velocity = np.random.normal(loc=0.0, scale=1e-3, size=self.n_variables)

    def get_next_point(self, start_point, n_epochs=250):
        new_point = [0 for _ in range(self.n_variables)]
        # velocity = np.random.normal(loc=0.0, scale=0.01, size=self.n_variables)
        for i in range(n_epochs):
            for j in range(self.n_variables):
                if self.loss == n_d_rastrigin:
                    self.velocity[j] = self.momentum * self.velocity[j] - self.learning_rate * self.loss_gradients[j](
                        start_point, j)
                    new_point[j] = start_point[j] + self.velocity[j]
                else:
                    self.velocity[j] = self.momentum * self.velocity[j] - self.learning_rate * self.loss_gradients[j](
                        *start_point)
                    new_point[j] = start_point[j] + self.velocity[j]
        if self.loss == n_d_rastrigin:
            return new_point, self.loss(new_point)
        return new_point, self.loss(*new_point)


class RMSProp(GradientDescent):
    def __init__(self, loss, loss_gradients, learning_rate=0.01, decay_rate=0.9, delta=1e-6):
        super().__init__(loss, loss_gradients, learning_rate)
        self.decay_rate = decay_rate
        self.delta = delta

    def get_next_point(self, start_point, n_epochs=250):
        new_point = [0 for _ in range(self.n_variables)]
        r = np.zeros(shape=self.n_variables)
        for i in range(n_epochs):
            for j in range(self.n_variables):
                if self.loss == n_d_rastrigin:
                    r[j] = self.decay_rate * r[j] + (1 - self.decay_rate) * self.loss_gradients[j](start_point, j) ** 2
                    delta_theta = -1 * (self.learning_rate / math.sqrt(self.delta + r[j])) * self.loss_gradients[j](
                        start_point, j)
                    new_point[j] = start_point[j] + delta_theta
                else:
                    r[j] = self.decay_rate * r[j] + (1 - self.decay_rate) * self.loss_gradients[j](*start_point) ** 2
                    delta_theta = -1 * (self.learning_rate / math.sqrt(self.delta + r[j])) * self.loss_gradients[j](
                        *start_point)
                    new_point[j] = start_point[j] + delta_theta
        if self.loss == n_d_rastrigin:
            return new_point, self.loss(new_point)
        return new_point, self.loss(*new_point)


class Adam(GradientDescent):
    def __init__(self, loss, loss_gradients, learning_rate=0.01, decay_rate_1=0.9, decay_rate_2=1e-6, delta=1e-8):
        super().__init__(loss, loss_gradients, learning_rate)
        self.decay_rate_1 = decay_rate_1
        self.decay_rate_2 = decay_rate_2
        self.delta = delta

    def get_next_point(self, start_point, n_epochs=250):
        new_point = [0 for _ in range(self.n_variables)]
        r = np.zeros(shape=self.n_variables)
        s = np.zeros(shape=self.n_variables)
        t = 0
        for i in range(n_epochs):
            t += 1
            for j in range(self.n_variables):
                if self.loss == n_d_rastrigin:
                    s[j] = self.decay_rate_1 * s[j] + (1 - self.decay_rate_1) * self.loss_gradients[j](start_point, j)
                    r[j] = self.decay_rate_2 * r[j] + (1 - self.decay_rate_2) * self.loss_gradients[j](start_point,
                                                                                                       j) ** 2
                    s_hat = s[j] / (1 - self.decay_rate_1 ** t)
                    r_hat = r[j] / (1 - self.decay_rate_2 ** t)
                    delta_theta = -1 * (self.learning_rate * s_hat / (self.delta + math.sqrt(r_hat))) * \
                                  self.loss_gradients[
                                      j](start_point, j)
                    new_point[j] = start_point[j] + delta_theta
                else:
                    s[j] = self.decay_rate_1 * s[j] + (1 - self.decay_rate_1) * self.loss_gradients[j](*start_point)
                    r[j] = self.decay_rate_2 * r[j] + (1 - self.decay_rate_2) * self.loss_gradients[j](
                        *start_point) ** 2
                    s_hat = s[j] / (1 - self.decay_rate_1 ** t)
                    r_hat = r[j] / (1 - self.decay_rate_2 ** t)
                    delta_theta = -1 * (self.learning_rate * s_hat / (self.delta + math.sqrt(r_hat))) * \
                                  self.loss_gradients[
                                      j](*start_point)
                    new_point[j] = start_point[j] + delta_theta
        if self.loss == n_d_rastrigin:
            return new_point, self.loss(new_point)
        return new_point, self.loss(*new_point)


class NewtonMethod(object):
    def __init__(self, loss, loss_gradients, loss_hessian, gamma=1e-2):
        self.gamma = gamma
        self.loss = loss
        self.loss_gradients = loss_gradients
        self.loss_hessian = loss_hessian
        self.n_variables = len(loss_gradients)

    def get_next_point(self, start_point, n_epochs=250):
        new_point = [0 for _ in range(self.n_variables)]
        for i in range(n_epochs):
            for j in range(self.n_variables):
                if self.loss == n_d_rastrigin:
                    new_point[j] = start_point[j] - self.gamma * np.linalg.inv(self.loss_hessian(start_point))[j, j] * \
                                   self.loss_gradients[j](start_point, j)
                else:
                    new_point[j] = start_point[j] - self.gamma * np.linalg.inv(self.loss_hessian(*start_point))[j, j] * \
                                   self.loss_gradients[j](*start_point)
        if self.loss == n_d_rastrigin:
            return new_point, self.loss(new_point)
        return new_point, self.loss(*new_point)


def visualize_optimization(loss_values, alg_name=None):
    plt.close("all")
    plt.figure(figsize=(15, 10))
    plt.title("Optimization Loss Convergence Comparison")
    for key, value in loss_values.items():
        plt.plot(value, label=key)
    # plt.xticks(np.arange(0, self.n_epochs + step, step), rotation=90)
    # plt.yticks(np.arange(0, results["train_loss"].max() + 2000, 2000))
    plt.ylabel("Loss")
    plt.xlabel("Time")
    plt.title(alg_name)
    plt.grid()
    plt.legend(loc="best")
    plt.savefig("./results/2/" + alg_name)


def run_optimization(opt_algorithm, start_point):
    point = start_point
    loss_values = []
    for i in range(20000):
        point, loss_value = opt_algorithm.get_next_point(point, n_epochs=1)
        loss_values.append(loss_value)
    return loss_values


if __name__ == '__main__':
    for loss_name in ["Rastrigin", "Ackley", "Levi", "Bukin", "n_D_Rastrigin"]:
        if loss_name == "Rastrigin":
            point = (0.5, 0.5)
            loss = rastrigin
            loss_gradients = [rastrigin_gradient_x, rastrigin_gradient_y]
            loss_hessian = rastrigin_hessian
        elif loss_name == "Ackley":
            point = (0.5, 0.5)
            loss = ackley
            loss_gradients = [ackley_gradient_x, ackley_gradient_y]
            loss_hessian = ackley_hessian
        elif loss_name == "Levi":
            point = (1.2, 1.2)
            loss = levi
            loss_gradients = [levi_gradient_x, levi_gradient_y]
            loss_hessian = levi_hessian
        elif loss_name == "Bukin":
            point = (-9.8, 0.8)
            loss = bukin
            loss_gradients = [bukin_gradient_x, bukin_gradient_y]
            loss_hessian = bukin_hessian
        elif loss_name == "n_D_Rastrigin":
            point = (0.5, 0.5, 0.5, 0.5)
            loss = n_d_rastrigin
            loss_gradients = [n_d_rastrigin_gradient, n_d_rastrigin_gradient, n_d_rastrigin_gradient,
                              n_d_rastrigin_gradient]
            loss_hessian = n_d_rastrigin_hessian
        loss_values = {}
        GDs = {}
        Nests = {}
        RMSs = {}
        Adams = {}
        NMs = {}
        for OptimizationAlgorithm in ["GradientDescent", "Nesterov", "RMSProp", "Adam", "NewtonMethod"]:
            for lr in [1e-3, 1e-5, 1e-6]:
                if OptimizationAlgorithm == "GradientDescent":
                    opt_alg = GradientDescent(loss, loss_gradients, learning_rate=lr)
                    GDs[loss_name + "_GradientDescent_lr=%f" % lr] = run_optimization(opt_alg, point)
                    visualize_optimization(GDs, loss_name + "_GradientDescentComparison.pdf")
                    print(loss_name + "_GradientDescent_lr=%f" % lr + " Finished!")
                elif OptimizationAlgorithm == "Nesterov":
                    for momentum in [0.7, 0.8, 0.9]:
                        opt_alg = Nesterov(loss, loss_gradients, learning_rate=lr, momentum=momentum)
                        Nests[loss_name + "_Nesterov_lr=%f_momentum=%.1f" % (lr, momentum)] = run_optimization(opt_alg, point)
                        visualize_optimization(Nests, loss_name + "_NesterovComparison.pdf")
                        print(loss_name + "_Nesterov_lr=%f_momentum=%.1f" % (lr, momentum) + " Finished!")
                elif OptimizationAlgorithm == "RMSProp":
                    for decay in [0.7, 0.8, 0.9]:
                        for delta in [1e-6, 1e-7, 1e-8]:
                            opt_alg = RMSProp(loss, loss_gradients, learning_rate=lr, decay_rate=decay, delta=delta)
                            RMSs[loss_name + "_RMSProp_lr=%f_decay=%.1f_delta=%.8f" % (lr, decay, delta)] = run_optimization(
                                opt_alg, point)
                            visualize_optimization(RMSs, loss_name + "_RMSPropComparison.pdf")
                            print(loss_name + "_RMSProp_lr=%f_decay=%.1f_delta=%.8f" % (lr, decay, delta) + " Finished")
                elif OptimizationAlgorithm == "Adam":
                    for d1 in [0.8, 0.9]:
                        for d2 in [0.95, 0.99]:
                            opt_alg = Adam(loss, loss_gradients, learning_rate=lr, decay_rate_1=d1, decay_rate_2=d2,
                                           delta=1e-6)
                            Adams[
                                loss_name + "_Adam_lr=%f_decay1=%.1f_decay2=%.2f_delta=%f" % (lr, d1, d2, 1e-6)] = run_optimization(
                                opt_alg, point)
                            visualize_optimization(Adams, loss_name + "_AdamComparison.pdf")
                            print(loss_name + "_Adam_lr=%f_decay1=%.1f_decay2=%.2f_delta=%f" % (lr, d1, d2, 1e-6) + " Finished")
                elif OptimizationAlgorithm == "NewtonMethod":
                    for gamma in [1e-2, 1e-3, 1e-4]:
                        opt_alg = NewtonMethod(loss, loss_gradients, loss_hessian, gamma=gamma)
                        NMs[loss_name + "_NewtonMethod_gamma=%f" % gamma] = run_optimization(opt_alg, point)
                        visualize_optimization(NMs, loss_name + "_NewtonMethodComparison.pdf")
                        print(loss_name + "_NewtonMethod_gamma=%f" % gamma + " Finished!")

        for OptimizationAlgorithm in ["GradientDescent", "Nesterov", "RMSProp", "Adam", "NewtonMethod"]:
            if OptimizationAlgorithm == "GradientDescent":
                opt_alg = GradientDescent(loss, loss_gradients, learning_rate=1e-5)
                loss_values["GradientDescent"] = run_optimization(opt_alg, point)
            elif OptimizationAlgorithm == "Nesterov":
                opt_alg = Nesterov(loss, loss_gradients, learning_rate=1e-5, momentum=0.8)
                loss_values["Nesterov"] = run_optimization(opt_alg, point)
            elif OptimizationAlgorithm == "RMSProp":
                opt_alg = RMSProp(loss, loss_gradients, learning_rate=1e-3, decay_rate=0.8, delta=1e-6)
                loss_values["RMSProp"] = run_optimization(opt_alg, point)
            elif OptimizationAlgorithm == "Adam":
                opt_alg = Adam(loss, loss_gradients, learning_rate=1e-3, decay_rate_1=0.9, decay_rate_2=0.99,
                               delta=1e-6)
                loss_values["Adam"] = run_optimization(opt_alg, point)
            elif OptimizationAlgorithm == "NewtonMethod":
                if loss_name == "Bukin":
                    continue
                opt_alg = NewtonMethod(loss, loss_gradients, loss_hessian, gamma=1e-4)
                loss_values["NewtonMethod"] = run_optimization(opt_alg, point)
        visualize_optimization(loss_values, loss_name + "_OptimizationsComparsion.pdf")
        print(loss_name + " Comparison of All Optimization Algorithms are finished!")
        print("All plots are saved in ./results/2/")
