import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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

    targets = targets.reshape(-1,1)
    return features, targets



if __name__ == '__main__':
    #Getting ours features and targets
    features, targets = get_dataset()

    #get a good visualisation
    #plt.scatter(features[:,0], features[:,1],s=40,  c=np.array(targets).flatten(), cmap=plt.cm.Spectral)
    #plt.show()

    #Initilization of the features and the targets, "None" means thats we can enter as features as we want
    tf_features = tf.placeholder(tf.float32, shape=[None,2])
    tf_targets = tf.placeholder(tf.float32, shape=[None,1])

    ###FIRST LAYER###
    #Initilization of the weight for THREE neurons, 2 features so [2,3]
    w1 = tf.Variable(tf.random_normal([2,300]))
    #Initilization of the weight for THREE neurons,1 bias per neuron so [3]
    b1 = tf.Variable(tf.zeros([300]))
    #Calculus of the pre activation
    z1 = tf.matmul(tf_features, w1) + b1
    #Calculus of the activation
    y1 = tf.nn.sigmoid(z1)
    #Calculus of the cost

    ###SECOND LAYER (OUTPUT)###
    #Initilization of the weight for ONE neuron, 3 features from the hidden layer so [2,1]
    w2 = tf.Variable(tf.random_normal([300,1]))
    #Initilization of the weight for ONE neuron,1 bias per neuron so [1]
    b2 = tf.Variable(tf.zeros([1]))
    #Calculus of the pre activation
    z2 = tf.matmul(y1, w2) + b2
    #Calculus of the activation
    y2 = tf.nn.sigmoid(z2)
    #Calculus of the cost



    cost = tf.reduce_mean(tf.square(y2 - tf_targets))
    #is the prediction correct

    correct_pred = tf.equal(tf.round(y2), tf_targets)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #OPTIMIZER
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train = optimizer.minimize(cost)

    #Initilization of the sessions and insert variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(50000):

        sess.run(train, feed_dict = {
            tf_features : features,
            tf_targets : targets
        })

        if epoch % 1000 == 0:
            print ("accuracy = ", sess.run(accuracy, feed_dict = {
                tf_features : features,
                tf_targets : targets
            }))
