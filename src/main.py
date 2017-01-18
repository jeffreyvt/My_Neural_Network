import mnist_loader as ml
import network_backprop_vectorised as network


if __name__ == "__main__":
    training_data, validation_data, test_data = ml.load_data_wrapper()


    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 0.1, test_data=test_data)