import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
    Created by Mohsen Naghipourfar on 10/10/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def delta_x(p):
    if p > 0:
        return 1
    else:
        return -1


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def train_test_split(x_data, y_data, test_split=0.20):
    n_samples = x_data.shape[0]
    train_size = int((1 - test_split) * n_samples)
    x_train, y_train = x_data.iloc[:train_size, :], y_data.iloc[:train_size]
    x_test, y_test = x_data.iloc[train_size:, :], y_data.iloc[train_size:]
    return x_train, x_test, y_train, y_test


def manipulate_data(data):
    # My Student ID is 94106757 -> Classifier will predict 5, 6
    new_data = data.loc[data[16].isin([5, 6]), :]
    new_data.loc[new_data[16] == 5, 16] = -1
    new_data.loc[new_data[16] == 6, 16] = 1
    return new_data


class Perceptron(object):
    def __init__(self, n_features, learning_rate=1e-6, logger_path=None, question_number=1):
        self.n_features = n_features
        self.weights = np.random.normal(loc=0.0, scale=1, size=(n_features,))
        self.learning_rate = learning_rate
        self.logger_path = logger_path
        self.question_number = question_number
        self.n_epochs = 1000
        print("The Perceptron Cell has been Created Successfully!")
        print("Initial Weights are:\t", end="")
        print(self.weights)
        print("Learning Rate :", self.learning_rate)
        print("*" * 50)

    def loss_1(self, x_train, y_train):
        loss = np.zeros(shape=(self.n_features,))
        n_samples = x_train.shape[0]
        for i in range(n_samples):
            p = self.linear_comb(x_train.iloc[i])
            if p * y_train.iloc[i] <= 0:
                loss += delta_x(p) * x_train.iloc[i]
        return loss

    def loss_2(self, x_train, y_train):
        loss = np.zeros(shape=(self.n_features,))
        n_samples = x_train.shape[0]
        for i in range(n_samples):
            p = self.linear_comb(x_train.iloc[i])
            if p * y_train.iloc[i] <= 0:
                loss += x_train.iloc[i] * ((p * (1 - math.tanh(p) ** 2)) + (math.tanh(p)))
        return loss

    def loss_3_1(self, x_train, y_train):
        c = 1.0
        loss = np.zeros(shape=(self.n_features,))
        n_samples = x_train.shape[0]
        for i in range(n_samples):
            p = self.linear_comb(x_train.iloc[i])
            if p * y_train.iloc[i] <= 0:
                if abs(p) > 1:
                    loss += c * x_train.iloc[i] * sign(p)
                else:
                    loss += x_train.iloc[i] * p
        return loss

    def loss_3_2(self, x_train, y_train):
        c = 0.001
        loss = np.zeros(shape=(self.n_features,))
        n_samples = x_train.shape[0]
        for i in range(n_samples):
            p = self.linear_comb(x_train.iloc[i])
            if p * y_train.iloc[i] <= 0:
                loss += x_train.iloc[i] * p / (math.sqrt(c + p ** 2))
        return loss

    def evaluate(self, x_test, y_test):
        n_samples = x_test.shape[0]
        n_correct_predictions = 0
        for i in range(n_samples):
            p = self.linear_comb(x_test.iloc[i])
            if p * y_test.iloc[i] > 0:
                n_correct_predictions += 1
        return n_correct_predictions / n_samples

    def fit(self, x_train, y_train, validation_set=(), n_epochs=250, verbose=0):
        self.n_epochs = n_epochs
        x_test, y_test = validation_set
        if os.path.exists(self.logger_path):
            os.remove(self.logger_path)
        for epoch in range(n_epochs):
            if self.question_number == 1:
                loss = self.loss_1(x_train, y_train)
            elif self.question_number == 2:
                loss = self.loss_2(x_train, y_train)
            elif self.question_number == 31:
                loss = self.loss_3_1(x_train, y_train)
            elif self.question_number == 32:
                loss = self.loss_3_2(x_train, y_train)
            self.weights -= self.learning_rate * loss
            train_loss = self.mean_squared_error(x_train, y_train)
            test_loss = self.mean_squared_error(x_test, y_test)
            train_acc = self.evaluate(x_train, y_train)
            test_acc = self.evaluate(x_test, y_test)
            if verbose:
                if validation_set:
                    print(
                        "Epoch-%d:\tTrain-Loss: %.4f\t\tTrain-Accuracy: %.4f\t\tTest-Loss: %.4f\t\tTest-Accuracy: %.4f" % (
                            epoch, train_loss, 100 * train_acc, test_loss, 100 * test_acc))
                    if self.logger_path:
                        with open(self.logger_path, 'a') as file:
                            writer = csv.writer(file)
                            if epoch == 0:
                                writer.writerow(["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])
                            writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])

    def linear_comb(self, X):
        return np.matmul(self.weights.transpose(), X)

    def mean_squared_error(self, x_data, y_data):
        loss = 0.0
        n_samples = x_data.shape[0]
        for i in range(n_samples):
            p = self.linear_comb(x_data.iloc[i])
            if p * y_data.iloc[i] <= 0:
                loss += (p - y_data.iloc[i]) ** 2
        return loss / n_samples

    def plot_results(self):
        step = 25
        results = pd.read_csv(self.logger_path)
        plt.close("all")
        plt.figure(figsize=(15, 10))
        plt.plot(results["epoch"], results["train_loss"], label="Training Loss")
        plt.plot(results["epoch"], results["test_loss"], label="Test Loss")
        plt.title("Perceptron's Loss for Question-%d" % self.question_number)
        plt.xticks(np.arange(0, self.n_epochs + step, step), rotation=90)
        plt.yticks(np.arange(0, results["train_loss"].max() + 2000, 2000))
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid()
        plt.legend(loc="best")
        plt.savefig("../Results/loss-%d.pdf" % self.question_number)
        plt.close("all")
        plt.figure(figsize=(15, 10))
        plt.plot(results["epoch"], 100 * results["train_accuracy"], label="Training Accuracy")
        plt.plot(results["epoch"], 100 * results["test_accuracy"], label="Test Accuracy")
        plt.xticks(np.arange(0, self.n_epochs + step, step), rotation=90)
        plt.yticks(np.arange(0, 105, 5))
        plt.title("Perceptron's Accuracy for Question-%d" % self.question_number)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.grid()
        plt.legend(loc="best")
        plt.savefig("../Results/acc-%d.pdf" % self.question_number)

    def print_confusion_matrix_latex(self, x_test, y_test):
        n_samples = x_test.shape[0]
        n_positive_samples = 0
        n_negative_samples = 0
        n_correct_positive_predictions = 0
        n_correct_negative_predictions = 0
        for i in range(n_samples):
            p = self.linear_comb(x_test.iloc[i])
            if y_test.iloc[i] > 0:
                n_positive_samples += 1
            else:
                n_negative_samples += 1
            if p * y_test.iloc[i] > 0:
                if p > 0:
                    n_correct_positive_predictions += 1
                else:
                    n_correct_negative_predictions += 1
        true_positive = n_correct_positive_predictions
        false_negative = n_positive_samples - n_correct_positive_predictions
        false_positive = n_negative_samples - n_correct_negative_predictions
        true_negative = n_correct_negative_predictions
        print("\t\t\tPredicted Class\t\t\t")
        print("\tP\t\tN\t")
        print("P\t%d\t\t%d\t" % (true_positive, false_negative))
        print("N\t%d\t\t%d\t" % (false_positive, true_negative))


if __name__ == '__main__':
    print("loading data...")
    data = pd.read_csv("../Data/digits.csv", header=None)
    data = manipulate_data(data)
    print("data has been loaded successfully.")
    print("data's shape is ", data.shape)
    x_data = data.drop([data.shape[1] - 1], axis=1)
    y_data = data[data.shape[1] - 1]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print("*" * 50)
    for q_num in [1, 2, 31, 32]:
        perceptron = Perceptron(n_features=x_data.shape[1], learning_rate=1e-6,
                                logger_path="../Results/log-%d.csv" % q_num,
                                question_number=q_num)
        perceptron.fit(x_train, y_train, validation_set=(x_test, y_test), n_epochs=1000, verbose=1)
        perceptron.print_confusion_matrix(x_test, y_test)
        perceptron.plot_results()
