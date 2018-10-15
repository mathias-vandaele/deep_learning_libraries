import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from scipy import ndimage
from scipy import misc
from random import randint


def create_conv(prev, filter_size, nb):
    conv_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, int(prev.get_shape()[-1]), nb)))
    conv_b = tf.Variable(tf.zeros(nb))
    conv   = tf.nn.conv2d(prev, conv_W, strides=[1, 1, 1, 1], padding='SAME') + conv_b
    # Activation: relu
    conv = tf.nn.relu(conv)
    # Pooling
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return conv

#qsdsqd
if __name__ == '__main__':
    targets = []
    features = []
    files = glob.glob("train/*jpg")
    random.shuffle(files)

    for file in files:
        features.append(np.array(Image.open(file).resize((75,75))))
        target = [1,0] if "cat" in file else [0,1]
        targets.append(target)

    features = np.array(features)
    targets = np.array(targets)

    """
    for a in [random.randint(0, len(features)) for _ in range(2)]:
        plt.imshow(features[a])
        plt.show()
    """

    X_train, X_valid, y_train , y_valid = train_test_split(features, targets, test_size=0.1, random_state=42)

    """
    print ("X_train", X_train.shape)
    print ("y_train", y_train.shape)
    print ("X_valid", X_valid.shape)
    print ("y_valid", y_valid.shape)
    """

    x = tf.placeholder(tf.float32, (None, 75, 75, 3) , name="image")
    y = tf.placeholder(tf.float32, (None, 2) , name="image")

    # Placeholder
    x = tf.placeholder(tf.float32, (None, 75, 75, 3), name="x")
    y = tf.placeholder(tf.float32, (None, 2), name="y")
    dropout = tf.placeholder(tf.float32, (None), name="dropout")

    conv = create_conv(x, 8, 32)
    conv = create_conv(conv, 5, 64)
    conv = create_conv(conv, 5, 128)
    conv = create_conv(conv, 5, 256)

    print (conv.shape)

    flat = flatten(conv)
    print(flat, flat.get_shape()[1])

    # First fully connected layer
    fc1_W = tf.Variable(tf.truncated_normal(shape=(int(flat.get_shape()[1]), 512)))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1   = tf.matmul(flat, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Last layer: Prediction
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, 2)))
    fc3_b  = tf.Variable(tf.zeros(2))
    logits = tf.matmul(fc1, fc3_W) + fc3_b

    softmax = tf.nn.softmax(logits)

    # Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)

    # Accuracy
    predicted_cls = tf.argmax(softmax, axis=1)
    correct_prediction = tf.equal(predicted_cls, tf.argmax(y, axis=1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
    training_operation = optimizer.minimize(loss_operation)


    #TRAINING PART
    batch_size = 100

    sess =  tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(100):
        index = np.arange(len(X_train))
        np.random.shuffle(index)
        X_train = X_train[index]
        y_train = y_train[index]

        for b in range(0, len(X_train), batch_size):
            batch = X_train[b:b+batch_size]
            y_train_batch = y_train[b:b+batch_size]
            sess.run(training_operation, feed_dict = {
                dropout: 0.8,
                x: batch,
                y: y_train_batch
            })


        acc = sess.run(accuracy_operation, feed_dict = {
            x: X_valid,
            y: y_valid
        })

        print ("Accuracy = ", np.mean(acc))
