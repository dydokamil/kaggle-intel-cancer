import random

from resize_all import resize_all
from tools import load_images, init_paths, RAW_DATA_PATH, PROCESSED_DATA_PATH, display_img, rotate_image, distort_image
import tensorflow as tf

RESIZED = True  # switch to False if images are not resized

if __name__ == '__main__':
    if not RESIZED:
        train_paths, valid_paths, test_paths = init_paths(RAW_DATA_PATH)
        # resize_all(train_paths, PROCESSED_DATA_PATH)
        # resize_all(valid_paths, PROCESSED_DATA_PATH)
        resize_all(test_paths, PROCESSED_DATA_PATH, labels=False)
    train_paths, valid_paths, test_paths = init_paths(PROCESSED_DATA_PATH)
    X_train, y_train = load_images(train_paths, small_dataset=True)
    test_img = random.sample(X_train, 1)[0]
    display_img(test_img)
    # rotated_sample = rotate_image(test_img)
    # display_img(rotated_sample)
    # display_img(rotated_sample)
    distorted_image = distort_image(test_img)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        display_img(sess.run(distorted_image))
