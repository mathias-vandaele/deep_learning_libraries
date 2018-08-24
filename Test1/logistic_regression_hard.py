"""
    Mathias Vandaele code
    code splited to the maximum in order to understand better machine learning
"""
import numpy as np
import matplotlib.pyplot as plt


def init_variables():
    """
        Init model variables (weights, bias)
    """
    weights_11 = np.random.normal(size=2)
    weights_12 = np.random.normal(size=2)
    weights_13 = np.random.normal(size=2)
    weights_output = np.random.normal(size=3)

    bias_11 = 0
    bias_12 = 0
    bias_13 = 0
    bias_output = 0
    return weights_11, weights_12, weights_13, weights_output, bias_11, bias_12, bias_13, bias_output

def get_dataset():
    """
        Method used to generate the dataset
    """
    #Number of rows per class
    row_per_class = 100
    #generate rows
    sick_people =  (np.random.randn(row_per_class,2)) + np.array([-2,-2])
    sick_people2 =  (np.random.randn(row_per_class,2)) + np.array([2,2])
    healthy_people = (np.random.randn(row_per_class,2)) + np.array([-2,2])
    healthy_people2 =  (np.random.randn(row_per_class,2)) + np.array([2,-2])

    features = np.vstack([sick_people,sick_people2, healthy_people, healthy_people2])
    targets = np.concatenate((np.zeros(row_per_class*2), np.zeros(row_per_class*2)+1))

    #features = np.vstack([[-2.9151944, -0.97887012],[ 1.64677955, 1.43885558],[-1.64697269,  0.68374247],[ 1.837042, -2.3003927 ]])
    #targets = [0., 0., 1., 1.]

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
        Only if we send z (NOT y !!!!)
    """
    return activation(z) * (1 - activation(z))

def derivative_activation_y(y):
    """
        compute the derivative of the activation
        IF we send the input after activation !!
    """
    return y * (1 - y)


def cost(predictions, targets):
    """
        make the difference between predictions and results
    """
    return np.mean((predictions - targets)**2)

def predict_hidden_layer(features, weights_11, weights_12, weights_13, bias_11, bias_12, bias_13):
    """
        This function is not generic at all and aims to understand how is made the input for the next ouput neural
    """
    predictions_11 = activation(pre_activation(features, weights_11, bias_11))
    predictions_12 = activation(pre_activation(features, weights_12, bias_12))
    predictions_13 = activation(pre_activation(features, weights_13, bias_13))
    layer1_result = np.stack((predictions_11, predictions_12, predictions_13), axis=1)
    return layer1_result



def predict_output_neural(features, weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output):
    """
        Determine the prediction of the output
    """
    layer1_result = predict_hidden_layer(features, weights_11, weights_12, weights_13, bias_11, bias_12, bias_13)
    #print (layer1_result)
    output_result = activation(pre_activation(layer1_result, weight_ouput, bias_output))
    #print (output_result)
    return layer1_result, output_result


def train_multiple_neurals(features, targets, weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output):
    """
        function of training multiple neural (ajust weights and bias in function of features and targets)
        This function is not generic or optimized and aims to understand better how it works
    """
    epochs = 10000
    learning_rate = 1

    #display Accuracy before the training
    layer1, prediction = predict_output_neural(features, weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output)
    predictions = np.around(prediction)
    print ("Accuracy", np.mean(predictions == targets))

    for epoch in range(epochs):
        layer1, predictions = predict_output_neural(features, weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output)
        if epoch % 10 == 0:
            layer1, predictions = predict_output_neural(features, weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output)
            print (cost(predictions, targets))
        """
            There are a lot of things to do here !
            to do the back propagation, we will first train the ouput neural
        """
        #Init gradient
        weights_gradient_output = np.zeros(weight_ouput.shape)
        bias_gradient_output = 0

        weights_gradient_11 = np.zeros(weights_11.shape)
        bias_gradient_11 = 0

        weights_gradient_12 = np.zeros(weights_12.shape)
        bias_gradient_12 = 0

        weights_gradient_13 = np.zeros(weights_12.shape)
        bias_gradient_13 = 0
        #Go throught each row
        for neural_input, feature, target, prediction in zip(layer1, features, targets, predictions):

            output_error = prediction - target
            output_delta = output_error * derivative_activation_y(prediction)

            error_neural_hidden_11 = output_delta * weight_ouput[0]
            error_neural_hidden_12 = output_delta * weight_ouput[1]
            error_neural_hidden_13 = output_delta * weight_ouput[2]


            error_neural_11 = error_neural_hidden_11 * derivative_activation_y(neural_input[0])
            error_neural_12 = error_neural_hidden_12 * derivative_activation_y(neural_input[1])
            error_neural_13 = error_neural_hidden_13 * derivative_activation_y(neural_input[2])

            weights_gradient_output += neural_input * output_delta
            #bias_output += output_delta

            weights_gradient_11 += feature * error_neural_11
            #bias_11 += error_neural_11

            weights_gradient_12 += feature * error_neural_12
            #bias_12 += error_neural_12

            weights_gradient_13 += feature * error_neural_13
            #bias_13 += error_neural_13


        #Update the weights and bias
        weight_ouput = weight_ouput - (learning_rate * weights_gradient_output)
        bias_output = bias_output - (learning_rate * bias_gradient_output)
        weights_11 =  weights_11 - (learning_rate * weights_gradient_11)
        bias_11 =  bias_11 - (learning_rate * bias_gradient_11)
        weights_12 =  weights_12 - (learning_rate * weights_gradient_12)
        bias_12 =  bias_12 - (learning_rate * bias_gradient_12)
        weights_13 =  weights_13 - (learning_rate * weights_gradient_13)
        bias_13 =  bias_13 - (learning_rate * bias_gradient_13)

    layer1, prediction = predict_output_neural(features, weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output)
    predictions = np.around(prediction)
    print ("Accuracy", np.mean(predictions == targets))


if __name__ == '__main__':
    #dataset
    features, targets  = get_dataset()
    #variables
    weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output = init_variables()
    #layer1_result, output_result = predict_output_neural(features,weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output)
    train_multiple_neurals(features, targets, weights_11, weights_12, weights_13, weight_ouput, bias_11, bias_12, bias_13, bias_output)
