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


def sample_image():
    sample_dir = random.sample(dirs, 1)[0]
    sample = random.sample(os.listdir(os.path.join(TRAIN_PATH, sample_dir)), 1)[0]
    sample = os.path.join(TRAIN_PATH, sample_dir, sample)
    img = plt.imread(sample)
    return img


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


class ImageQueue:
    def __init__(self, paths, min_queue_examples=5):
        self.__paths = paths
        self.__queue = []
        self.__min_queue_examples = min_queue_examples
        self.__enqueue_thread = threading.Thread(target=self.__auto_enqueue)
        self.__enqueue_thread.start()

    def __auto_enqueue(self):
        while len(self.__queue) < self.__min_queue_examples:
            self.__enqueue_example()
        # print('Enqueued examples. Current size:', len(self.__queue))

    def __input_producer(self):
        path = random.sample(self.__paths, 1)[0]
        img = plt.imread(path)
        self.__queue.append([img, path.split('/')[-2]])

    def __enqueue_example(self):
        self.__input_producer()

    def dequeue_example(self, size=1, consume=True):
        if self.__enqueue_thread:
            self.__enqueue_thread.join()
        if consume:
            batch = self.__queue[0:size]
            del self.__queue[0:size]
            # maybe append new elements after they've been deleted
            # print(f"Dequeueing {size} examples")
            # print(f"Current size: {len(self.__queue)}")
            self.__enqueue_thread = threading.Thread(target=self.__auto_enqueue)
            self.__enqueue_thread.start()
            return batch
        else:
            return self.__queue[0:size]


train_paths, valid_paths = init_paths()
image_queue = ImageQueue(train_paths)
# print('Waiting, then dequeueing one example')
image_queue.dequeue_example()
# for i in range(3):
#     image_queue.enqueue_example()
#
# for i in range(3):
#     print(image_queue.dequeue_example())
# input_producer(train_paths)
# img, label = input_producer(train_paths)
# plt.imshow(img)
# plt.show()
