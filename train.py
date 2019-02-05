import os

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

DATA_DIR = 'data'
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_DATA = 'train_data.npy'
TRAIN_LABELS = 'train_labels.npy'
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
TEST_DATA = 'test_data.npy'
TEST_LABELS = 'test_labels.npy'

MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'model.h5')

BATCH_SIZE = 32


if __name__ == '__main__':
    train_data = np.load(os.path.join(TRAIN_DATA_DIR, TRAIN_DATA)) / 255.0
    train_labels = np.load(os.path.join(TRAIN_DATA_DIR, TRAIN_LABELS))

    test_data = np.load(os.path.join(TEST_DATA_DIR, TEST_DATA)) / 255.0
    test_labels = np.load(os.path.join(TEST_DATA_DIR, TEST_LABELS))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(60, 80, 1)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(6))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=1, validation_data=(test_data, test_labels))
    model.save(MODEL_FILE)
