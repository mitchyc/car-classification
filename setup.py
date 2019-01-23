import os
import sys

import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm

DATA_DIR = 'data'

TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
DEVKIT_DIR = os.path.join(DATA_DIR, 'devkit')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_CAR_DIR = os.path.join(TRAIN_DATA_DIR, 'car')
TRAIN_NO_CAR_DIR = os.path.join(TRAIN_DATA_DIR, 'nocar')
TEST_CAR_DIR = os.path.join(TEST_DATA_DIR, 'car')
TEST_NO_CAR_DIR = os.path.join(TEST_DATA_DIR, 'nocar')

TRAIN_DATA_NPY = os.path.join(TRAIN_DATA_DIR, 'train_data.npy')
TRAIN_LABELS_NPY = os.path.join(TRAIN_DATA_DIR, 'train_labels.npy')

TEST_DATA_NPY = os.path.join(TEST_DATA_DIR, 'test_data.npy')
TEST_LABELS_NPY = os.path.join(TEST_DATA_DIR, 'test_labels.npy')

WIDTH = 80
HEIGHT = 60
CHANNELS = 1


def resize_image(im, size, points):
    # Resize an image
    # Crops and pads an image while maintaining original aspect ratio
    aspect = im.size[0] / im.size[1]
    # x1, y1, width, height
    points = (points[0] / im.size[0], points[1] / im.size[1],
              (points[2] - points[0]) / im.size[0], (points[3] - points[1]) / im.size[1])
    im = im.resize((size[0], int(size[0] / aspect)), Image.ANTIALIAS)
    diff_y = size[1] - im.size[1]
    pad_y = int(diff_y / 2)
    if diff_y > 0:
        im_pad = Image.new('L', size)
        im_pad.paste(im, (0, pad_y))
        im = im_pad
    elif diff_y < 0:
        im = im.crop((0, -pad_y, size[0], size[1] - pad_y))
    return im, points


if __name__ == '__main__':
    # Load training annotations and car classes
    train_annos = scipy.io.loadmat(os.path.join(DEVKIT_DIR, 'cars_train_annos.mat'))['annotations']
    test_annos = scipy.io.loadmat(os.path.join(DEVKIT_DIR, 'cars_test_annos.mat'))['annotations']
    # classes = scipy.io.loadmat(os.path.join(DEVKIT_DIR, 'cars_meta.mat'))

    train_car_images = os.listdir(TRAIN_CAR_DIR)
    train_no_car_images = os.listdir(TRAIN_NO_CAR_DIR)
    test_car_images = os.listdir(TEST_CAR_DIR)
    test_no_car_images = os.listdir(TEST_NO_CAR_DIR)

    train_car_image_count = len(train_annos[0])
    test_car_image_count = len(test_annos[0])
    train_no_car_image_count = len(train_no_car_images)
    test_no_car_image_count = len(test_no_car_images)

    train_image_count = train_car_image_count + train_no_car_image_count
    test_image_count = test_car_image_count + test_no_car_image_count

    # Load/process images in training set
    batch_no = 0
    train_data = np.empty((train_image_count, HEIGHT, WIDTH, CHANNELS))
    train_labels = np.zeros((train_image_count, 6), dtype='float32')
    i = 0
    for ann in tqdm(train_annos[0], desc='Processing training images (car)', file=sys.stdout, unit='images'):
        im = Image.open(os.path.join(TRAIN_CAR_DIR, ann[5][0]))
        im = im.convert('L')
        im, points = resize_image(im, (WIDTH, HEIGHT), (ann[0][0], ann[1][0], ann[2][0], ann[3][0]))

        train_data[i] = np.expand_dims(np.array(im), -1)
        train_labels[i][0] = 1
        train_labels[i][2:] = points
        i += 1

    for no_car in tqdm(train_no_car_images, desc='Processing training images (no car)', file=sys.stdout, unit='images'):
        im = Image.open(os.path.join(TRAIN_NO_CAR_DIR, no_car))
        im = im.convert('L')
        im, _ = resize_image(im, (WIDTH, HEIGHT), (0, 0, 0, 0))

        train_data[i] = np.expand_dims(np.array(im), -1)
        train_labels[i][1] = 1

        i += 1

    p = np.random.permutation(len(train_data))
    train_data = train_data[p]
    train_labels = train_labels[p]

    np.save(TRAIN_DATA_NPY, train_data)
    np.save(TRAIN_LABELS_NPY, train_labels)

    # Load/process images in test set
    batch_no = 0
    test_data = np.empty((test_image_count, HEIGHT, WIDTH, CHANNELS))
    test_labels = np.zeros((test_image_count, 6), dtype='float32')
    i = 0
    for ann in tqdm(test_annos[0], desc='Processing test images (car)', file=sys.stdout, unit='images'):
        im = Image.open(os.path.join(TEST_CAR_DIR, ann[4][0]))
        im = im.convert('L')
        im, points = resize_image(im, (WIDTH, HEIGHT), (ann[0][0], ann[1][0], ann[2][0], ann[3][0]))

        test_data[i] = np.expand_dims(np.array(im), -1)
        test_labels[i][0] = 1
        test_labels[i][2:] = points
        i += 1

    for no_car in tqdm(test_no_car_images, desc='Processing test images (no car)', file=sys.stdout, unit='images'):
        im = Image.open(os.path.join(TEST_NO_CAR_DIR, no_car))
        im = im.convert('L')
        im, _ = resize_image(im, (WIDTH, HEIGHT), (0, 0, 0, 0))

        test_data[i] = np.expand_dims(np.array(im), -1)
        test_labels[i][1] = 1

        i += 1

    p = np.random.permutation(len(test_data))
    test_data = test_data[p]
    test_labels = test_labels[p]

    np.save(TEST_DATA_NPY, test_data)
    np.save(TEST_LABELS_NPY, test_labels)
