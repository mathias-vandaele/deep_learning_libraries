"""
    Simple code made to understand easily the gradient descent
"""

if __name__ == '__main__':
    #function to minimize
    fc = lambda x, y : (3*x**2) + (x*y) + (5*y**2)
    #Set partial variables
    partial_derivative_x = lambda x, y : (6*x) + y
    partial_derivative_y = lambda x, y : (10*y) + x
    #Set variables
    x = 10
    y = -13

    #Set learning rate
    learning_rate = 0.01
    #One epoch is one period of minimization
    for epoch in range(0,2000):
        #Compute gradient
        x_gradient = partial_derivative_x(x,y)
        y_gradient = partial_derivative_y(x,y)
        #Apply gradient descent
        x = x - learning_rate * x_gradient
        y = y - learning_rate * y_gradient
        #Keep track of the function
        print ("Fc = %s" % fc(x,y))

    print ("")
    print ("x = %s" % x)
    print ("y = %s" % y)
