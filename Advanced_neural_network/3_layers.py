"""
    author : VANDAELE Mathias
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

def sigmoid_derivative(y):
    return y * (1.0 - y)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],20)
        self.weights2   = np.random.rand(20,20)
        self.weights3   = np.random.rand(20,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.bias1      = np.random.rand(1,20)
        self.bias2      = np.random.rand(1,20)
        self.bias3      = np.random.rand(1,1)
        self.learning_rate = 0.01

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.bias1)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        self.output = sigmoid(np.dot(self.layer2, self.weights3) + self.bias3)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights2 =  np.dot(self.layer1.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
        d_weights1 = np.dot(self.input.T,  (np.dot(np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1)))

        d_bias3 = 2*(self.y - self.output) * sigmoid_derivative(self.output)
        d_bias3 = d_bias3.mean(axis=0)

        d_bias2    = np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)
        d_bias2    = d_bias2.mean(axis=0)

        d_bias1    = np.dot(np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1)
        d_bias1    = d_bias1.mean(axis=0)


        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1 * self.learning_rate
        self.weights2 += d_weights2 * self.learning_rate
        self.weights3 += d_weights3 * self.learning_rate
        self.bias1    += d_bias1 * self.learning_rate
        self.bias2    += d_bias2 * self.learning_rate
        self.bias3    += d_bias3 * self.learning_rate

    def cost(self):
        return np.mean((self.output - self.y)**2)


if __name__ == "__main__":


    """
    row_per_class = 160
    #Creating a data set hard to resolve
    sick_people =  (np.random.randn(row_per_class,2))
    row_sick = int(row_per_class/8)
    healthy_people =  2*(np.random.randn(row_sick,2)) + np.array([0,10])
    healthy_people2 = 2*(np.random.randn(row_sick,2)) + np.array([0,-10])
    healthy_people3 = 2*(np.random.randn(row_sick,2)) + np.array([10,0])
    healthy_people4 = 2*(np.random.randn(row_sick,2)) + np.array([-10,0])
    healthy_people5 =  2*(np.random.randn(row_sick,2)) + np.array([10,10])
    healthy_people6 = 2*(np.random.randn(row_sick,2)) + np.array([10,-10])
    healthy_people7 = 2*(np.random.randn(row_sick,2)) + np.array([-10,10])
    healthy_people8 = 2*(np.random.randn(row_sick,2)) + np.array([-10,-10])
    features = np.vstack([sick_people, healthy_people2, healthy_people, healthy_people3, healthy_people4, healthy_people5, healthy_people6, healthy_people7, healthy_people8])
    targets = (np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class)+1)))

    #To have a good vision of the dataset created just above
    plt.scatter(features[:,0], features[:,1], c=targets, cmap = plt.cm.Spectral)
    plt.show()
    targets = targets[np.newaxis].T

    """

    #Number of rows per class
    row_per_class = 100
    #generate rows
    sick_people =  (np.random.randn(row_per_class,2)) + np.array([-2,-2])
    sick_people2 =  (np.random.randn(row_per_class,2)) + np.array([2,2])
    healthy_people = (np.random.randn(row_per_class,2)) + np.array([-2,2])
    healthy_people2 =  (np.random.randn(row_per_class,2)) + np.array([2,-2])

    features = np.vstack([sick_people,sick_people2, healthy_people, healthy_people2])
    targets = (np.concatenate((np.zeros(row_per_class*2), np.zeros(row_per_class*2)+1)))

    #To have a good vision of the dataset created just above
    plt.scatter(features[:,0], features[:,1], c=targets, cmap = plt.cm.Spectral)
    plt.show()
    targets = targets[np.newaxis].T



    nn = NeuralNetwork(features,targets)



    nn.feedforward()
    predictions = np.around(nn.output)
    print ("Accuracy", np.mean(predictions == nn.y))

    for i in range(30000):
        if i  % 1000 == 0:
            print (nn.cost())
        nn.feedforward()
        nn.backprop()

    nn.feedforward()
    predictions = np.around(nn.output)
    print ("Accuracy", np.mean(predictions == nn.y))

    predictions = np.around(np.squeeze(np.asarray(nn.output)))
    plt.scatter(features[:,0], features[:,1], c=predictions, cmap = plt.cm.Spectral)
    plt.show()

    row_per_class = 2000
    #generate rows
    sick_people =  (np.random.randn(row_per_class,2))*4
    sick_people2 =  (np.random.randn(row_per_class,2))*4
    healthy_people = (np.random.randn(row_per_class,2))*4
    healthy_people2 =  (np.random.randn(row_per_class,2))*4
    features = np.vstack([sick_people,sick_people2, healthy_people, healthy_people2])

    nn.input = features
    nn.feedforward()

    predictions = np.around(np.squeeze(np.asarray(nn.output)))
    plt.scatter(features[:,0], features[:,1], c=predictions, cmap = plt.cm.Spectral)
    plt.show()
