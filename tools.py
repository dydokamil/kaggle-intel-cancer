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


def input_producer(train_paths):
    pass


train_paths, valid_paths = init_paths()
