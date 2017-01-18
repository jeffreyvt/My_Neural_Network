import numpy as np
import random


class Network:

    def __init__(self, sizes):
        """
        :param sizes: contains the number of neurons in the respective
        layers of the network. For example, if the list was [1, 2, 3],
        then the first layer will have 1 neuron, second layer will have
        2 neurons, and the third layer will have 3 neurons.

        The biases and the weights of the neurons are initiated randomly
        using a Gaussian distribution with mean 0 and variance 1. i.e.
        np.random.randn(arraysize)

        note: the first layer is considered to be the input layer, hence
        by contention we won't be setting any biases for those neurons,
        since the biases are only ever used in computing the outputs from
        later layers.

        note: for random samples from N(mu, sigma^2), use:
        sigma * np.random.randn(arraysize) + mu
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        :param a: is the input for the neural network
        :return: the output of the neural network
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        :param training_data:
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param test_data:
        :return:
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaulate(test_data), n_test
                ))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        This function updates the network's weight and biases by
        applying gradient descent using backpropagation to a single
        mini_batch.
        :param mini_batch: a list of tuples (x, y)
        :param eta: learning rate
        :return: None
        """
        # initialise nabla_b and nabla_w
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # apply magical function
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
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
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        print(delta.shape)
        print(activations[-2].transpose().shape)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        print(nabla_w[-1])
        input()
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

    def evaulate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)



    def cost_derivative(self, output_activation, y):
        """
        :param output_activation: self explanatory
        :param y: self explanatory
        :return: returns the dc/da of the quadratic cost function
        """
        return output_activation - y

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