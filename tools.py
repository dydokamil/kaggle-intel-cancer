import threading
import os
from PIL import Image
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

RAW_DATA_PATH = r"/media/hdd/training_data/intel-cancer/Screening"
SHORTER_AXIS = 227

PROCESSED_DATA_PATH = '/home/kamil/training_data/intel-cancer/resized'


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
        while len(self.__queue) < self.__min_queue_examples:
            self.__enqueue_example()

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
            self.__enqueue_thread = threading.Thread(target=self.__auto_enqueue)
            self.__enqueue_thread.start()
            return batch
        else:
            return self.__queue[0:size]


def resize_img(img):
    img = Image.fromarray(img)
    multiplier = 1
    if img.size[0] > img.size[1]:
        img = img.rotate(90)
        # multiplier = img.size[1] / SHORTER_AXIS
    elif img.size[1] > img.size[0]:
        multiplier = img.size[0] / SHORTER_AXIS
    new_size = (round(img.size[0] / multiplier), round(img.size[1] / multiplier))
    img.thumbnail(new_size)
    return img


def load_images(paths, test_dataset=False, small_dataset=False):
    i = 0
    images, labels = [], []
    for path in paths:
        try:
            img = plt.imread(path)
            img = img.astype(np.float32)
            images.append(img)
            if not test_dataset:
                string_label = path.split('/')[-2]
                if string_label == "Type_1":
                    labels.append([0, 0, 1])
                elif string_label == "Type_2":
                    labels.append([0, 1, 0])
                elif string_label == "Type_3":
                    labels.append([1, 0, 0])
            i = i + 1
        except Exception as e:
            print(e)
        if i == 500 and small_dataset:
            break
    if test_dataset:
        return images
    else:
        return np.asarray(images), np.asarray(labels)


def rotate_image(img, angle):
    img = Image.fromarray(img)
    return img.rotate(random.randint(0, angle))


def normalize(img):
    return tf.divide(img, 255.)


def distort_image(img):
    img = tf.image.random_brightness(img, .1)
    img = tf.image.random_contrast(img, .9, 1.)
    img = tf.image.random_hue(img, .01)
    img = tf.image.random_saturation(img, .9, 1.)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    return img


def next_batch(images, labels, distort=True, random_shuffle=False, batch_size=None):
    if random_shuffle:
        indices = random.sample(range(len(images)), batch_size)
        raw_images = images[indices]
        labels = labels[indices]
    else:
        raw_images = images
    processed_images = raw_images
    if distort:
        processed_images = [distort_image(image) for image in processed_images]
    processed_images = [normalize(image) for image in processed_images]
    return processed_images, labels


def resize_all(queue, output_path, labels=True):
    for image_path in queue:
        name = image_path.split('/')[-1]
        if labels:
            cancer_type = image_path.split('/')[-2]
        try:
            image = plt.imread(image_path)
            resized = resize_img(image)
            if labels:
                train_or_valid = image_path.split('/')[-3]
                resized.save(os.path.join(output_path, train_or_valid, cancer_type, name))
            else:
                resized.save(os.path.join(output_path, 'test', name))
        except Exception as e:
            print(f'Wonky image: {image_path}; Skipping...')
            continue
