import numpy as np
import tensorflow as tf
import os

from tools import load_images, init_paths, RAW_DATA_PATH, PROCESSED_DATA_PATH, next_batch, resize_all

RESIZED = True  # switch to False if images are not resized
NUM_EPOCHS = 50000
BATCH_SIZE = 5
LOSS_LOG_AFTER = 15
MODEL_SAVE_PATH = '/home/kamil/deep-learning/saved-models/intel-cancer'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)


x = tf.placeholder(tf.float32, shape=[None, None, None, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

# first convolutional layer
W_conv1 = weight_variable([11, 11, 3, 96])
b_conv1 = bias_variable([96])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, [1, 4, 4, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

lrn1 = tf.nn.local_response_normalization(h_pool1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 96, 256])
b_conv2 = bias_variable([256])

h_conv2 = tf.nn.relu(tf.nn.conv2d(lrn1, W_conv2, [1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

lrn2 = tf.nn.local_response_normalization(h_pool2)

# third convolutional layer
W_conv3 = weight_variable([3, 3, 256, 384])
b_conv3 = bias_variable([384])
h_conv3 = tf.nn.relu(tf.nn.conv2d(lrn2, W_conv3, [1, 1, 1, 1], padding='SAME') + b_conv3)

# fourth convolutional layer
W_conv4 = weight_variable([3, 3, 384, 384])
b_conv4 = bias_variable([384])
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, [1, 1, 1, 1], padding='SAME') + b_conv4)

# fifth convolutional layer
W_conv5 = weight_variable([3, 3, 384, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, [1, 1, 1, 1], padding='SAME') + b_conv5)
h_pool5 = tf.nn.max_pool(h_conv5, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

lrn3 = tf.nn.local_response_normalization(h_pool5)

# sixth convolutional layer
keep_prob = tf.placeholder(tf.float32)
W_conv6 = weight_variable([1, 1, 256, 4096])
b_conv6 = bias_variable([4096])
h_conv6 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(lrn3, W_conv6, [1, 1, 1, 1], padding='VALID') + b_conv6),
                        keep_prob=keep_prob)
lrn3 = tf.nn.local_response_normalization(h_conv6)

W_conv7 = weight_variable([1, 1, 4096, 4096])
b_conv7 = bias_variable([4096])
h_conv7 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(lrn3, W_conv7, [1, 1, 1, 1], padding='VALID') + b_conv7),
                        keep_prob=keep_prob)
lrn4 = tf.nn.local_response_normalization(h_conv7)

W_conv8 = weight_variable([1, 1, 4096, 3])
b_conv8 = bias_variable([3])
# h_conv8 = tf.nn.softmax(tf.nn.conv2d(h_conv7, W_conv8, [1, 1, 1, 1], padding='VALID') + b_conv8)
h_conv8 = tf.nn.conv2d(lrn4, W_conv8, [1, 1, 1, 1], padding='VALID') + b_conv8

avg_pool = tf.reduce_mean(h_conv8, axis=1)
avg_pool2 = tf.reduce_mean(avg_pool, axis=1)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=avg_pool2)
# loss = tf.subtract(avg_pool2, y_)
train_step = tf.train.AdamOptimizer().minimize(loss)

if __name__ == '__main__':
    previous_loss = 500.
    if not RESIZED:
        train_paths, valid_paths, test_paths = init_paths(RAW_DATA_PATH)
        resize_all(test_paths, PROCESSED_DATA_PATH, labels=False)
    train_paths, valid_paths, test_paths = init_paths(PROCESSED_DATA_PATH)
    X_valid, y_valid = load_images(valid_paths)
    X_train, y_train = load_images(train_paths)
    all_losses = []

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        X_valid_batch_op, y_valid_batch = next_batch(X_valid, y_valid, distort=False)
        for i in range(NUM_EPOCHS):
            # check accuracy
            if i % LOSS_LOG_AFTER == 0:
                loss_valid = sess.run([loss], feed_dict={x: X_valid_batch_op,
                                                         y_: y_valid_batch,
                                                         keep_prob: 1.})
                loss_valid = np.average(loss_valid)
                print(f"Validation loss: {loss_valid}")
                if loss_valid < previous_loss:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, 'my-model'))
                    previous_loss = loss_valid
            # train
            X_train_batch_op, y_train_batch = next_batch(X_train,
                                                         y_train,
                                                         distort=False,
                                                         random_shuffle=True,
                                                         batch_size=BATCH_SIZE)
            sess.run([train_step], feed_dict={x: X_train_batch_op,
                                              y_: y_train_batch,
                                              keep_prob: .5})
