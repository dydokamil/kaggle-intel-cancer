import os
import random

import matplotlib.pyplot as plt

DATA_PATH = r"/media/hdd/training_data/intel-cancer/Screening"
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')


def sample_image():
    dirs = ['Type_1', 'Type_2', 'Type_3']
    sample_dir = random.sample(dirs, 1)[0]
    sample = random.sample(os.listdir(os.path.join(TRAIN_PATH, sample_dir)), 1)[0]
    sample = os.path.join(TRAIN_PATH, sample_dir, sample)
    img = plt.imread(sample)
    plt.imshow(img)
    plt.show()


sample_image()
