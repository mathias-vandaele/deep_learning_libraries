import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    #change logging settings
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)


    mnist = input_data.read_data_sets("input/data", one_hot=True)
    print (mnist.train.labels.shape)

    """
    plt.imshow(mnist.train.images[0].reshape(28,28), cmap="gray")
    plt.show()
    """

    #Image de 784 pixels
    tf_features = tf.placeholder(tf.float32, shape=[None,784])
    tf_targets = tf.placeholder(tf.float32, shape=[None,10])

    #VARIABLES
    w1 = tf.Variable(tf.random_normal([784,10]))
    b1 = tf.Variable(tf.zeros([10]))
    #PRE ACTIVATION
    z1 = tf.matmul(tf_features, w1) + b1
    #SOFTMAX
    y1 = tf.nn.softmax(z1)

    ##ERROR
    error = tf.nn.softmax_cross_entropy_with_logits(labels = tf_targets, logits=z1)
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(error)

    ##PREDICTION VERIFICATION
    correct_pred = tf.equal(tf.argmax(y1,1), tf.argmax(tf_targets,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        """
        sess.run(train, feed_dict = {
            tf_features : [mnist.train.images[0]],
            tf_targets : [mnist.train.labels[0]]
        })"""

        epochs = 5000

        #TRAINING
        for epoch in range(epochs):

            batch_features, batch_targets = mnist.train.next_batch(1000)

            sess.run(train, feed_dict = {
                tf_features : batch_features,
                tf_targets : batch_targets
                })

            if epoch % 100 == 0:
                print (sess.run(accuracy, feed_dict = {
                    tf_features : batch_features,
                    tf_targets : batch_targets
                }))


        print (sess.run(accuracy, feed_dict = {
            tf_features : mnist.test.images,
            tf_targets : mnist.test.labels
        }))


    #Reset old logging settings
    tf.logging.set_verbosity(old_v)
