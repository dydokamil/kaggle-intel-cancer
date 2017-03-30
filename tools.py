import threading
import os
from PIL import Image
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

DATA_PATH = r"/media/hdd/training_data/intel-cancer/Screening"
dirs = ['Type_1', 'Type_2', 'Type_3']
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')
VALID_PATH = os.path.join(DATA_PATH, 'valid')
SHORTER_AXIS = 227


def init_paths():
    train_paths = []
    valid_paths = []
    for dir in dirs:
        type_path_train = os.path.join(TRAIN_PATH, dir)
        type_path_valid = os.path.join(VALID_PATH, dir)
        files_train = os.listdir(type_path_train)
        files_valid = os.listdir(type_path_valid)
        train_paths.extend([os.path.join(type_path_train, file) for file in files_train])
        valid_paths.extend([os.path.join(type_path_valid, file) for file in files_valid])
    return train_paths, valid_paths


def display_img(img):
    plt.imshow(img)
    plt.show()


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
