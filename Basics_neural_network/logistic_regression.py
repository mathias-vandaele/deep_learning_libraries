"""
    small code made to understand how a single neural works, that implements the notion of logistic regression
    and gradient descent
"""
import numpy as np
import matplotlib.pyplot as plt


def init_variables():
    """
        Init model variables (weights, biais)
    """
    weights = np.random.normal(size=2)
    bias = 0
    return weights, bias

def get_dataset():
    """
        Method used to generate the dataset
    """
    #Number of rows per class
    row_per_class = 100
    #generate rows
    sick_people =  (np.random.randn(row_per_class,2)) + np.array([-2,-2])
    healthy_people = (np.random.randn(row_per_class,2)) + np.array([2,2])

    features = np.vstack([sick_people, healthy_people])
    targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class)+1))

    plt.scatter(features[:,0], features[:,1], c=targets, cmap = plt.cm.Spectral)
    plt.show()

    return features, targets

def pre_activation(features, weights, bias):
    """
        compute pre activation of the neural
    """
    return np.dot(features, weights) + bias

def activation(z):
    """
        compute the activation (sigmoide)
    """
    return 1 / ( 1 + np.exp(-z) )

def derivative_activation(z):
    """
        compute the derivative of the activation (derivative of sigmoide)
    """
    return activation(z) * (1 - activation(z))

def predict(features, weights, bias):
    """
        aims to make the prediction
    """
    #compute pre activation
    z = pre_activation(features, weights, bias)
    #compute the activation
    y = activation(z)
    #return the round of the prediction (0 or 1)
    return np.round(y)

def cost(predictions, targets):
    """
        make the difference between predictions and results
    """
    return np.mean((predictions - targets)**2)

def train(features, targets, weights, bias):
    """
        function of training (ajust weights and bias in function of features and targets)
    """
    epochs = 100
    learning_rate = 0.1

    #Afficher les points
    #plt.scatter(features[:,0], features[:,1], c=targets, cmap = plt.cm.Spectral)
    #plt.show()

    #display Accuracy before the training
    predictions = predict(features, weights, bias)
    print ("Accuracy", np.mean(predictions == targets))

    for epoch in range(epochs):
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print (cost(predictions, targets))

        #Init gradient
        weights_gradient = np.zeros(weights.shape)
        bias_gradient = 0
        #Go throught each row
        for feature, target in zip(features, targets):
            #compute pre activation
            z = pre_activation(feature, weights, bias)
            #compute the activation
            y = activation(z)
            #Update the gradient
            weights_gradient += (y - target)* derivative_activation(z) * feature
            bias_gradient += (y - target)* derivative_activation(z)

        #Update the weights and bias
        weights = weights - (learning_rate * weights_gradient)
        bias = bias - (learning_rate * bias_gradient)

    print ("final weights", weights)
    print ("final bias", bias)
    #Display the Accuracy after the training
    predictions = predict(features, weights, bias)
    print ("Accuracy", np.mean(predictions == targets))




if __name__ == '__main__':
    #dataset
    features, targets  = get_dataset()
    #variables
    weights, bias = init_variables()
    train(features, targets, weights, bias)
