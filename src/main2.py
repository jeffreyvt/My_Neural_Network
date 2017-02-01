import mnist_loader as ml
import network2 as network


if __name__ == "__main__":
    training_data, validation_data, test_data = ml.load_data_wrapper()


    net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.05,
            lmbda=1,
            evaluation_data=validation_data,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_training_cost=True)

    # #search for a good hyper parameter
    # lmbdas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
    # etas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
    #
    # best_accuracy = 0
    # best_hyper_parameter = "Not Found"
    # for lmbda in lmbdas:
    #     for eta in etas:
    #         print("eta = {}, lmbda = {}".format(eta, lmbda))
    #         evaluation_cost, evaluation_accuracy, \
    #         training_cost, training_accuracy = \
    #             net.SGD(training_data[:5000], 30, 10, eta,
    #                     lmbda=lmbda,
    #                     evaluation_data=validation_data[:500],
    #                     monitor_evaluation_accuracy=True)
    #         if evaluation_accuracy[-1] > best_accuracy:
    #             best_accuracy = evaluation_accuracy[-1]
    #             best_hyper_parameter = (lmbda, eta)
    # print(best_hyper_parameter)