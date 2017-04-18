import numpy as np
import tensorflow as tf
import os

import tflearn
from tflearn import input_data, conv_2d, max_pool_2d, local_response_normalization, fully_connected, dropout, regression

from tools import load_images, init_paths, RAW_DATA_PATH, PROCESSED_DATA_PATH, next_batch, resize_all

RESIZED = True  # switch to False if images are not resized
NUM_EPOCHS = 50000
BATCH_SIZE = 2000
LOSS_LOG_AFTER = 15
MODEL_SAVE_PATH = '/home/kamil/deep-learning/saved-models/intel-cancer'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)


x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)

if __name__ == '__main__':
    previous_loss = 500.
    if not RESIZED:
        train_paths, valid_paths, test_paths = init_paths(RAW_DATA_PATH)
        resize_all(test_paths, PROCESSED_DATA_PATH, labels=False)
    train_paths, valid_paths, test_paths = init_paths(PROCESSED_DATA_PATH)
    X_valid, y_valid = load_images(valid_paths)
    X_train, y_train = load_images(train_paths)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        X_valid_batch_op, y_valid_batch = next_batch(X_valid, y_valid, distort=False)
        for i in range(NUM_EPOCHS):
            # train
            X_train_batch_op, y_train_batch = next_batch(X_train,
                                                         y_train,
                                                         distort=False,
                                                         random_shuffle=True,
                                                         batch_size=BATCH_SIZE)
            model = tflearn.DNN(network)
            model.fit(X_train_batch_op, y_train_batch, n_epoch=1000, validation_set=0.1, shuffle=True,
                      show_metric=True, batch_size=20, snapshot_step=200,
                      snapshot_epoch=False, run_id='cancer')
