import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    #change logging settings
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)


    mnist = input_data.read_data_sets("input/data", one_hot=True)
    print (mnist.train.images.shape)

    #Reset old logging settings
    tf.logging.set_verbosity(old_v)
