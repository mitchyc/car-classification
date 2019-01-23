import os

import numpy as np
import tensorflow as tf
from PIL import Image

from setup import resize_image

EVAL_DIR = 'eval'

MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'model.h5')

HEIGHT = 60
WIDTH = 80
CHANNELS = 1

if __name__ == '__main__':
    if not os.path.isdir(EVAL_DIR):
        os.mkdir(EVAL_DIR)

    eval_files = os.listdir(EVAL_DIR)
    file_count = len(eval_files)
    eval_data = np.empty(shape=(file_count, HEIGHT, WIDTH, CHANNELS))
    i = 0
    for file in eval_files:
        im = Image.open(os.path.join(EVAL_DIR, file))
        im = im.convert('L')
        im, _ = resize_image(im, (WIDTH, HEIGHT), (0, 0, 0, 0))
        im = np.array(im)
        im = np.expand_dims(im, -1)
        eval_data[i] = im
        i += 1

    model = tf.keras.models.load_model(MODEL_FILE)
    predictions = model.predict(eval_data)
    print(predictions)
