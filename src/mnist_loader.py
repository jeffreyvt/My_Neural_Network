"""
This script loads the MNIST image data for the neural network
"""

import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    f = gzip.open("../data/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def show_plot(data):
    data = data.reshape(28,28)
    plt.imshow(data, cmap=plt.cm.binary)
    plt.show()


def load_data_wrapper():
    td, vd, tsd = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in td[0]]
    training_results = [vectorized_result(y) for y in td[1]]
    training_data = zip(training_inputs,training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in vd[0]]
    validation_results = [vectorized_result(x) for x in vd[1]]
    validation_data = zip(validation_inputs, validation_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in tsd[0]]
    test_data = zip(test_inputs, tsd[1])
    return list(training_data), list(validation_data), list(test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

