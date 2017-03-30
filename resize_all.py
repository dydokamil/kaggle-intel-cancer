OUTPUT_FOLDER = '/home/kamil/training_data/intel-cancer/resized'
import matplotlib.pyplot as plt
import os
from tools import init_paths, resize_img


def resize_queue(queue):
    for image_path in queue:
        cancer_type = image_path.split('/')[-2]
        name = image_path.split('/')[-1]
        if name == '.DS_Store':
            continue
        try:
            image = plt.imread(image_path)
            resized = resize_img(image)
            resized.save(os.path.join(OUTPUT_FOLDER, cancer_type, name))
        except Exception as e:
            print(f'Wonky image: {image_path}. Skipping...')
            continue


def resize_all():
    train_paths, valid_paths = init_paths()
    print(train_paths[0])
    resize_queue(train_paths)
    resize_queue(valid_paths)
