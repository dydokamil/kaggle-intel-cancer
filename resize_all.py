import matplotlib.pyplot as plt
import os
from tools import init_paths, resize_img


def resize_all(queue, output_path, labels=True):
    for image_path in queue:
        name = image_path.split('/')[-1]
        # if name == '.DS_Store':
        #     continue
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
