# My_Neural_Network
A basic Neural Network implementation in Python 3.5 used to recognise MNIST hand written digits. The code is inspired by the book: http://neuralnetworksanddeeplearning.com/. Modifications were made to vectorise the back propagation algorithm to speed up the speed of learning by at least 2x. The current [784, 30, 10] neural network architecture has achieved 95% accuracy with the current MNIST dataset. Further works has been dedicated in improving the network accuracy by reading further into the book.


# src/main.py
Executes the network.py using standard sigmoid cost function. This implementation is able to achieve ~ 95% accuracy

# src/main2.py
Executes the network.py using L2 regularised cross entropy cost function. This implementation is able to achieve > 96% accuracy. Hyper parameter optimization was also tested for choosing eta (the learning rate) and lambda (the regularization parameter). Current experimentaiton found that eta = 0.05 and lambda = 1 gives the best learning result

Further works includes implementing dropout to the neural networks to reduce the overfitting. Other methods will be explored in order to further improve the MNIST classification accuracy.
