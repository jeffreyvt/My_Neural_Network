"""
This is an improved version of the network.py, implementing the stochastic
gradient descent learning algorithm for feedforward neural network. Improvements
include the addition of the cross-entropy cost function, regularization, and
better initialization of network weights.
"""


import numpy as np
import random
import json
import sys

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """
        :param a: output
        :param y: desired output
        :return: cost associated with the input a and y
        """
        return 0.5 * np.linalg.norm(a-y) ** 2

    @staticmethod
    def delta(z, a, y):
        """
        :param z: input
        :param a: output
        :param y: desired output
        :return: the delta associated with the inputs
        """
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        :param a: output
        :param y: desired output
        :return: the cost associated with a and y
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """
        :param z: input
        :param a: output
        :param y: desired output
        :return: the delta associated with z, a and y
        """
        return a-y


class Network:

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        :param sizes: contains the number of neurons in the respective
        layers of the network. For example, if the list was [1, 2, 3],
        then the first layer will have 1 neuron, second layer will have
        2 neurons, and the third layer will have 3 neurons.

        The biases and the weights of the neurons are initiated randomly
        using a Gaussian distribution using the self.default_weight_initializer"
        This is one of the many tweaks to make the neural network learn
        faster and better than before

        note: the first layer is considered to be the input layer, hence
        by contention we won't be setting any biases for those neurons,
        since the biases are only ever used in computing the outputs from
        later layers.

        note: for random samples from N(mu, sigma^2), use:
        sigma * np.random.randn(arraysize) + mu
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        Initialize each weight using a Gausian Distribution with mean 0
        and standard deviation = 1/sqrt(n_in) where n_in is the number of
        weights connecting to the same neuron. Initialize the biases using
        a Gaussian Distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and by
        convention we won't set any biases for those neurons, since biases
        are only ever used in computing the outputs from later layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def large_weight_initializer(self):
        """
        Initialize the weights using the standard Gaussian distribution with
        mean = 0 and standard deviation = 1. Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer and by
        convention we won't set any biases for those neurons, since biases
        are only ever used in computing the output from later layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def feedforward(self, a):
        """
        :param a: is the input for the neural network
        :return: the output of the neural network
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        Train the neural network using mini-batch stochastic gradient descent.

        :param training_data: is a list of tuples (x, y) representing the training
        inputs and the desired outputs.
        :param lmbda: is the regularization parameter.

        The method also accepts "evaluation_data", which is usually the test data
        or the validation data.
        The cost and accuracy on the evaluation data or the training data ca be
        monitored using the appropriate flags made available in the params.
        The method returns a tuple containing four lists: the (per epoch) costs
        on the evaluation data, the accuracies on the evaluation data, the costs
        on the training data, and the accuracies on the training data.
        All values are evaluated at the end of each epoch.
        """
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy \
            = [], [], [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch {} training is complete".format(j))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}, {}%".format(accuracy, n, accuracy/n*100))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}, {}%"
                      .format(accuracy, n_data, accuracy/n_data*100))  # n_data is the size of the evaulation data
            print()
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        This function updates the network's weight and biases by
        applying gradient descent using backpropagation to a single
        mini_batch.
        :param mini_batch: a list of tuples (x, y)
        :param eta: learning rate
        :param lmbda: is the regularization parameter
        :param n: is the total size of the training data set
        :return: None
        """
        # initialise nabla_b and nabla_w
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """
        :param x: self explanatory
        :param y: self explanatory
        :return: returns the dc/db and dc/dw of the cost functions based
        on the input training example. The returned values are then used
        to train the neural network.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward pass
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        # print(nabla_b)
        # print(nabla_b[0].shape)
        # print(nabla_b[1].shape)
        # input()
        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        """

        :param data: the input data to check the accuracy of the neural network
        with. The data can be validation data, test data or training data.
        :param convert: should set to False if the data set is validation or
        test data, and to True if the data set is the training data. The need
        for this flag arises due to the difference in the way the results "y"
        are represented in the different data sets. see mnist_loader.load_data_wrapper
        :return: the number of inputs in "data" for which the neural network
        outputs the correct result. The neural network's output is assumed to
        be the index of whichever neuraon in the final layer has the highest
        activation.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                      for x, y in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                      for x, y in data]
        return sum(int(x == y) for x, y in results)





    def total_cost(self, data, lmbda, convert=False):
        """
        :param data: the data used for cost computation
        :param lmbda: the regularization parameter
        :param convert: flag should be set to False when dealing with training
        data, and set to True when dealing with validation or test data
        :return: The total cost for the data set "data".
        """
        cost = 0.0
        for x, y in data:
            # print(x)
            # input()
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            # print(a, y)
            # input()
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5 * (lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost


    def save(self, filename):
        """
        saves the neural network to the filename
        :param filename: the file name to save the neural network to
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__),
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    """
    load a neural network from the file "filename", return an instance of Network
    """
    f = open(filename, "r")
    data = json.load()
    f.close()
    cost = getattr(sys.module[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    """
    :param y: the classification of the text
    :return: vectorized representation of the classification (desired
    output from the neural network)
    """
    # print("lalala {}".format(j))
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """
    :param z: the input for the sigmoid function
    :return: output of the sigmoid function
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    :param z: the input for the derivative of the sigmoid function
    :return: the derivative of the sigmoid function at z
    """
    return sigmoid(z)*(1-sigmoid(z))