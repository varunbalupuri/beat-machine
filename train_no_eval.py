# train_with_tf.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import numpy as np
import pickle
import sys
from matplotlib import pyplot as plt

EPOCHS = 10
ITERATIONS = 500

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

def alex_layer(input_layer, filters, kernel_size, pool_size, strides):
    input_shape = input_layer.get_shape()
    print('input shape:', input_shape)

    conv = tf.layers.conv2d(
          inputs=input_layer,
          filters=filters,
          kernel_size=kernel_size,
          padding="same",
          activation=tf.nn.relu
          )

    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=pool_size, strides=strides)

    print('output:', pool.get_shape())

    return pool


def build_alex_net(features, labels, learning_rate=0.01):
    input_layer = tf.reshape(features, [-1, 64, 1022, 1])

    p1 = alex_layer(input_layer, 16, [5,5], [2,2], 2)

    p2 = alex_layer(p1, 16, [5,5], [2,2], 2)

    p3 = alex_layer(p2, 16, [5,5], [2,2], 2)

    p4 = alex_layer(p3, 16, [5,5], [2,2], 2)

    pool2_flat = tf.reshape(p4, [-1, 4*63*16])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 4*63*16]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dense2 = tf.layers.dense(inputs=dense, units=512, activation=tf.nn.relu)

    dense3 = tf.layers.dense(inputs=dense2, units=200, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense3, rate=0.1)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 13]

    logits = tf.layers.dense(inputs=dropout, units=13)

    # training operations

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss=loss)

    init = tf.global_variables_initializer()

    return init, logits, train_op, loss



def cnn_model_fn(features, labels, learning_rate=0.01):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 64x1022 pixels, and have one color channel
    input_layer = tf.reshape(features, [-1, 64, 1022, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 64, 1022, 1]
    # Output Tensor Shape: [batch_size, 64, 1022, 32]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    print(conv1.get_shape())

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 64, 1022, 32]
    # Output Tensor Shape: [batch_size, 32, 511, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 511, 32]
    # Output Tensor Shape: [batch_size, 32, 511, 64]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 32, 511, 64]
    # assuming floor is used....
    # Output Tensor Shape: [batch_size, 16, 255, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #### EXTRA LAYER

    # Convolutional Layer #3 TODO: change the filter sizes
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 16, 255, 64]
    # Output Tensor Shape: [batch_size, 16, 255, 128]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 16, 255, 128]
    # assuming floor is used....
    # Output Tensor Shape: [batch_size, 8, 127, 128]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 8, 127, 128]
    # Output Tensor Shape: [batch_size, 8*127*128]
    pool2_flat = tf.reshape(pool2, [-1, 16 * 255 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 8*127*128]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 13]

    logits = tf.layers.dense(inputs=dropout, units=13)

    # training operations

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss=loss)

    init = tf.global_variables_initializer()

    return init, logits, train_op, loss, pool2



def main(dataset_path):
    # Load training and eval data

    with open(dataset_path, 'rb') as handle:
        dataset = pickle.load(handle)

    train_data = np.array(dataset['X'])
    train_labels = np.array([target - np.array([123]) for target in dataset['y']])

    X = tf.placeholder(dtype=tf.float32, shape=[None, 64, 1022])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

    init, logits, train_op, loss = build_alex_net(X, y)

    batch_size = int(len(dataset['X'])/ITERATIONS)
    print('batch_size', batch_size)
    print()

    losses = []

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(EPOCHS):

            print('epoch:', epoch)
            for iteration in range(ITERATIONS):
                X_batch = train_data[iteration*batch_size:(iteration+1)*batch_size]
                y_batch = train_labels[iteration*batch_size:(iteration+1)*batch_size]

                loss_const, _ = sess.run([loss, train_op], feed_dict={X: X_batch, y: y_batch})

                losses.append(loss_const)
                print(loss_const)

    plt.plot(losses)

    plt.show()


if __name__ == "__main__":
    DATASET_PATH = 'datasets/mel_dataset_2018-04-02_len_5670.pickle'
    main(DATASET_PATH)