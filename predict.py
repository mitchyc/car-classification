import os

import numpy as np
import tensorflow as tf
from PIL import Image
from setup import resize_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.set_printoptions(suppress=True)

EVAL_DIR = 'eval'

MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'model.h5')

HEIGHT = 60
WIDTH = 80
CHANNELS = 1

THRESHOLD = 0

if __name__ == '__main__':
    eval_files = os.listdir(EVAL_DIR)
    file_count = len(eval_files)
    eval_data = np.empty(shape=(file_count, HEIGHT, WIDTH, CHANNELS))
    orig_data = [None] * file_count
    i = 0
    for file in eval_files:
        im = Image.open(os.path.join(EVAL_DIR, file))
        orig_data[i] = im
        im = im.convert('L')
        im, _ = resize_image(im, (WIDTH, HEIGHT), (0, 0, 0, 0))
        im = np.array(im) / 255.0
        im = np.expand_dims(im, -1)
        eval_data[i] = im
        i += 1

    model = tf.keras.models.load_model(MODEL_FILE)
    predictions = model.predict(eval_data)
    del model

    for i in range(file_count):
        if predictions[i][0] > THRESHOLD:
            p0 = predictions[i][2] * orig_data[i].size[0], predictions[i][3] * orig_data[i].size[1]
            box_size = (predictions[i][4] - predictions[i][2]) * orig_data[i].size[0], (
                    predictions[i][5] - predictions[i][3]) * orig_data[i].size[1]
            fig, ax = plt.subplots(1)
            ax.imshow(orig_data[i])
            box = patches.Rectangle(p0, box_size[0], box_size[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(box)
            plt.show()

            print('{}'.format(predictions[i]))
