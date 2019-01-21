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

# Raw image data
TRAIN_CAR_DIR = os.path.join(TRAIN_DATA_DIR, 'car')
TRAIN_NO_CAR_DIR = os.path.join(TRAIN_DATA_DIR, 'nocar')
TEST_CAR_DIR = os.path.join(TEST_DATA_DIR, 'car')
TEST_NO_CAR_DIR = os.path.join(TEST_DATA_DIR, 'nocar')

# Preprocessed image dirs
TRAIN_DATA_NPY_PREFIX = os.path.join(TRAIN_DATA_DIR, 'train_data')
TRAIN_LABELS_NPY = os.path.join(TRAIN_DATA_DIR, 'train_labels.npy')
TRAIN_BOXES_NPY = os.path.join(TRAIN_DATA_DIR, 'train_boxes.npy')

TEST_DATA_NPY_PREFIX = os.path.join(TEST_DATA_DIR, 'test_data')
TEST_LABELS_NPY = os.path.join(TEST_DATA_DIR, 'test_labels.npy')
TEST_BOXES_NPY = os.path.join(TEST_DATA_DIR, 'test_boxes.npy')

BATCH_SIZE = 32

WIDTH = 80
HEIGHT = 60
CHANNELS = 3


def get_relative_position(p1, p2, size):
    # Return the pixel percentage of the given points respective of the image size
    return (p1[0] / size[0], p1[1] / size[1]), (p2[0] / size[0], p2[1] / size[1])


def resize_image(im, size):
    # Resize an image
    # Crops and pads an image while maintaining original aspect ratio
    aspect = im.size[0] / im.size[1]
    im = im.resize((size[0], int(size[0] / aspect)), Image.ANTIALIAS)
    diff_y = size[1] - im.size[1]
    pad_y = int(diff_y / 2)
    if diff_y > 0:
        im_pad = Image.new('RGB', size, (0, 0, 0))
        im_pad.paste(im, (0, pad_y))
        im = im_pad
    elif diff_y < 0:
        im = im.crop((0, -pad_y, size[0], size[1] - pad_y))
    return im


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
    train_data = np.empty((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS))
    train_labels = np.empty((train_image_count,), dtype='int32')
    train_boxes = np.empty((train_image_count, 2, 2), dtype='int32')
    i = 0
    img = 0
    for ann in tqdm(train_annos[0], desc='Processing training images (car)', file=sys.stdout, unit='images'):
        im = Image.open(os.path.join(TRAIN_CAR_DIR, ann[5][0]))
        im = im.convert('RGB')
        im = resize_image(im, (WIDTH, HEIGHT))

        train_data[img] = np.array(im)
        train_labels[i] = 1

        # Convert bounding box pixel points to relative points
        p1, p2 = (int(ann[0][0]), int(ann[1][0])), (int(ann[2][0]), int(ann[3][0]))
        box = get_relative_position(p1, p2, im.size)
        train_boxes[i] = box
        i += 1
        img += 1

        if i % BATCH_SIZE == 0:
            np.save('{}_{}.npy'.format(TRAIN_DATA_NPY_PREFIX, batch_no), train_data)
            batch_no += 1
            img = 0
            train_data = np.empty((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS))

    for no_car in tqdm(train_no_car_images, desc='Processing training images (no car)', file=sys.stdout, unit='images'):
        im = Image.open(os.path.join(TRAIN_NO_CAR_DIR, no_car))
        im = im.convert('RGB')
        im = resize_image(im, (WIDTH, HEIGHT))

        train_data[img] = np.array(im)
        train_labels[i] = 0

        # Convert bounding box pixel points to relative points
        train_boxes[i] = ((0, 0), (0, 0))
        i += 1
        img += 1

        if i % BATCH_SIZE == 0:
            np.save('{}_{}.npy'.format(TRAIN_DATA_NPY_PREFIX, batch_no), train_data)
            batch_no += 1
            img = 0
            train_data = np.empty((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS))

    if img > 0:
        np.save('{}_{}.npy'.format(TRAIN_DATA_NPY_PREFIX, batch_no), train_data)

    np.save(TRAIN_LABELS_NPY, train_labels)
    np.save(TRAIN_BOXES_NPY, train_boxes)

    # Load/process images in test set
    batch_no = 0
    test_data = np.empty((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS))
    test_labels = np.empty((test_image_count,), dtype='int32')
    test_boxes = np.empty((test_image_count, 2, 2), dtype='int32')
    i = 0
    img = 0
    for ann in tqdm(test_annos[0], desc='Processing test images (car)', file=sys.stdout, unit='images'):
        im = Image.open(os.path.join(TEST_CAR_DIR, ann[4][0]))
        im = im.convert('RGB')
        im = resize_image(im, (WIDTH, HEIGHT))

        test_data[img] = np.array(im)
        test_labels[i] = 1

        # Convert bounding box pixel points to relative points
        p1, p2 = (int(ann[0][0]), int(ann[1][0])), (int(ann[2][0]), int(ann[3][0]))
        box = get_relative_position(p1, p2, im.size)
        test_boxes[i] = box
        i += 1
        img += 1

        if i % BATCH_SIZE == 0:
            np.save('{}_{}.npy'.format(TEST_DATA_NPY_PREFIX, batch_no), test_data)
            batch_no += 1
            img = 0
            test_data = np.empty((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS))

    for no_car in tqdm(test_no_car_images, desc='Processing test images (no car)', file=sys.stdout, unit='images'):
        im = Image.open(os.path.join(TEST_NO_CAR_DIR, no_car))
        im = im.convert('RGB')
        im = resize_image(im, (WIDTH, HEIGHT))

        test_data[img] = np.array(im)
        test_labels[i] = 0

        # Insert dummy points (no car)
        test_boxes[i] = ((0, 0), (0, 0))
        i += 1
        img += 1

        if i % BATCH_SIZE == 0:
            np.save('{}_{}.npy'.format(TEST_DATA_NPY_PREFIX, batch_no), test_data)
            batch_no += 1
            img = 0
            test_data = np.empty((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS))

    if img > 0:
        np.save('{}_{}.npy'.format(TEST_DATA_NPY_PREFIX, batch_no), test_data)

    np.save(TEST_LABELS_NPY, test_labels)
    np.save(TEST_BOXES_NPY, test_boxes)
