import threading
import os
from PIL import Image
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

RAW_DATA_PATH = r"/media/hdd/training_data/intel-cancer/Screening"
# dirs = ['Type_1', 'Type_2', 'Type_3']
# RAW_TRAIN_PATH = os.path.join(RAW_DATA_PATH, 'train')
# RAW_TEST_PATH = os.path.join(RAW_DATA_PATH, 'test')
# RAW_VALID_PATH = os.path.join(RAW_DATA_PATH, 'valid')
SHORTER_AXIS = 227

PROCESSED_DATA_PATH = '/home/kamil/training_data/intel-cancer/resized'


# PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_PATH, 'train')
# PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_PATH, 'test')
# PROCESSED_VALID_PATH = os.path.join(PROCESSED_DATA_PATH, 'valid')

def init_paths(main_path):
    train_paths = []
    test_paths = []
    valid_paths = []
    for root, dirs, files in os.walk(main_path):
        for file in files:
            if file.endswith('.jpg'):
                full_path = os.path.join(root, file)
                if '/test/' in full_path:
                    test_paths.append(full_path)
                elif '/valid/' in full_path:
                    valid_paths.append(full_path)
                elif '/train/' in full_path:
                    train_paths.append(full_path)
    return train_paths, valid_paths, test_paths


def display_img(img):
    plt.imshow(img)
    plt.show()
    plt.draw()


class ImageQueue:
    '''This class implements a threaded image queue.'''

    def __init__(self, paths, min_queue_examples=5, normalize=True):
        self.__paths = paths
        self.__normalize = normalize
        self.__queue = []
        self.__min_queue_examples = min_queue_examples
        self.__enqueue_thread = threading.Thread(target=self.__auto_enqueue)
        self.__enqueue_thread.start()
        print("Filling up the queue...")

    def __auto_enqueue(self):
        # print("Filling up the queue...")
        while len(self.__queue) < self.__min_queue_examples:
            self.__enqueue_example()
            # print('Filled up the queue. Current size:', len(self.__queue))

    def __input_producer(self):
        path = random.sample(self.__paths, 1)[0]
        img = plt.imread(path)
        self.__queue.append([img, path.split('/')[-2]])

    def __enqueue_example(self):
        self.__input_producer()

    def dequeue_example(self, size=1, consume=True):
        if size > self.__min_queue_examples:
            raise Exception("Requested amount of examples is greater than min_queue_example.")

        if self.__enqueue_thread:
            self.__enqueue_thread.join()

        if size > len(self.__queue):
            raise Exception("Requested amount of examples is greater than the size of the queue.")

        if consume:
            # TODO if queue_size == size then return queue
            batch = []
            for i in range(size):
                batch.append(self.__queue.pop())
            # maybe append new elements after they've been deleted
            # print(f"Dequeueing {size} examples")
            # print(f"Current size: {len(self.__queue)}")
            self.__enqueue_thread = threading.Thread(target=self.__auto_enqueue)
            self.__enqueue_thread.start()
            return batch
        else:
            return self.__queue[0:size]


def resize_img(img):
    img = Image.fromarray(img)
    multiplier = 1
    if img.size[0] > img.size[1]:  # width > height
        multiplier = img.size[1] / SHORTER_AXIS
    elif img.size[1] > img.size[0]:  # height > width
        multiplier = img.size[0] / SHORTER_AXIS
    new_size = (round(img.size[0] / multiplier), round(img.size[1] / multiplier))
    img.thumbnail(new_size)
    # print(new_size)
    return img


def load_images(paths, test_dataset=False, small_dataset=False):
    i = 0
    images, labels = [], []
    for path in paths:
        try:
            images.append(plt.imread(path))
            if not test_dataset:
                labels.append(path.split('/')[-2])
            i = i + 1
        except Exception as e:
            print(e)
        if i == 50 and small_dataset:
            break
    if test_dataset:
        return images
    else:
        return images, labels


def rotate_image(img, angle):
    # return tf.contrib.image.rotate(img, angle)
    img = Image.fromarray(img)
    return img.rotate(random.randint(0, angle))


def distort_image(img):
    img = tf.image.flip_left_right(img)
    img = tf.image.random_brightness(img, .1)
    img = tf.image.random_contrast(img, .9, 1.)
    img = tf.image.random_hue(img, .01)
    img = tf.image.random_saturation(img, .9, 1.)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.rot90(img, random.randint(0, 3))
    return img
