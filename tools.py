import os
import random
import threading

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

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


def load_images(paths, test_dataset=False):
    i = 0
    images, labels = [], []
    for path in paths:
        try:
            img = Image.open(path)
            # img = img.astype(np.float32)
            images.append(np.array(img))
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
    if test_dataset:
        return np.asarray(images)
    else:
        return np.asarray(images), np.asarray(labels)


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

    if batch_size is None:
        batch_size = len(raw_images)

    processed_images = []
    for image in raw_images:
        processed_images.append(normalize(np.array(Image.fromarray(image).resize([227, 227]))))

    # ensure squareness
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
