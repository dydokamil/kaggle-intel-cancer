import tensorflow as tf

from tools import load_images, init_paths, RAW_DATA_PATH, PROCESSED_DATA_PATH, next_batch, resize_all

RESIZED = True  # switch to False if images are not resized
NUM_EPOCHS = 1

if __name__ == '__main__':
    if not RESIZED:
        train_paths, valid_paths, test_paths = init_paths(RAW_DATA_PATH)
        resize_all(test_paths, PROCESSED_DATA_PATH, labels=False)
    train_paths, valid_paths, test_paths = init_paths(PROCESSED_DATA_PATH)
    X_train, y_train = load_images(train_paths, small_dataset=True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCHS):
            # X_train_batch_op, y_train_batch = next_batch(10, X_train, y_train)
            # images = sess.run(X_train_batch_op)
            # print(images)
            pass