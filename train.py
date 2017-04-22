import numpy as np
import tensorflow as tf

from tools import load_images, init_paths, RAW_DATA_PATH, PROCESSED_DATA_PATH, next_batch, resize_all

RESIZED = True  # switch to False if images are not resized
NUM_EPOCHS = 50000
BATCH_SIZE = 5
LOSS_LOG_AFTER = 15
MODEL_SAVE_PATH = '/home/kamil/deep-learning/saved-models/intel-cancer'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.5)
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

# ------------------------------fully connected------------------------------ #
keep_prob = tf.placeholder(tf.float32)
lrn3_flat = tf.reshape(lrn3, [-1, 7 * 7 * 256])

W_fc1 = weight_variable([7 * 7 * 256, 4096])
b_fc1 = bias_variable([4096])
h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(lrn3_flat, W_fc1) + b_fc1), keep_prob)

W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])
h_fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2), keep_prob)

W_fc3 = weight_variable([4096, 3])
b_fc3 = bias_variable([3])
y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
                # loss_valid, predictions = sess.run([cross_entropy, correct_prediction], feed_dict={x: X_valid_batch_op,
                #                                                                                    y_: y_valid_batch,
                #                                                                                    keep_prob: 1.})
                train_accuracy = accuracy.eval(feed_dict={x: X_valid_batch_op, y_: y_valid_batch, keep_prob: 1.})
                # print(f"Validation loss: {loss_valid}, accuracy: {train_accuracy}")
                print(f"Train accuracy: {train_accuracy}")
                # if loss_valid < previous_loss:
                #     saver.save(sess, os.path.join(MODEL_SAVE_PATH, 'my-model'))
                    # previous_loss = loss_valid
            # train
            X_train_batch_op, y_train_batch = next_batch(X_train,
                                                         y_train,
                                                         distort=False,
                                                         random_shuffle=True,
                                                         batch_size=BATCH_SIZE)
            sess.run([train_step], feed_dict={x: X_train_batch_op,
                                              y_: y_train_batch,
                                              keep_prob: .5})
