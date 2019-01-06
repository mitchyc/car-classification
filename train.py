import os
import sys
import tarfile

import Augmentor
import numpy as np
import requests
import scipy.io
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# Training/testing resource URL/file names
TRAIN_DATA_URL = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
DEVKIT_URL = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
TEST_DATA_URL = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'

TRAIN_DATA_FILE_NAME = TRAIN_DATA_URL.split('/')[-1]
TRAIN_DEVKIT_FILE_NAME = DEVKIT_URL.split('/')[-1]
TEST_DATA_FILE_NAME = TEST_DATA_URL.split('/')[-1]


def download_file(url, chunk_size=1024, progress_bar=True):
    # Download a file from the specified URL
    fname = url.split('/')[-1]
    fpath = './data/{}'.format(fname)
    r = requests.get(url, stream=True)
    total_size = int(r.headers["Content-Length"])
    bars = int(total_size / chunk_size)
    with open(fpath, "wb") as f:
        if progress_bar is True:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=bars, unit="KB",
                              desc=fname, file=sys.stdout):
                f.write(chunk)
        else:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)


def extract_tgz(fname, dest):
    # Extract the given tgz file
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(dest)
    tar.close()


def get_relative_position(p1, p2, size):
    # Return the pixel percentage of the given points respective of the image size
    return (p1[0] / size[0], p1[1] / size[1]), (p2[0] / size[0], p2[1] / size[1])


def resize_image(im, size=(800, 600)):
    # Resize an image (default is 800x600)
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
    if not os.path.isdir('./data'):
        os.mkdir('data')

    if not os.path.isdir('./data/devkit/') or not os.listdir('./data/devkit/'):
        if not os.path.isfile('./data/{}'.format(TRAIN_DEVKIT_FILE_NAME)):
            # Download devkit archive
            print('Retrieving training annotations...')
            download_file(DEVKIT_URL, chunk_size=32, progress_bar=False)
        # Extract devkit training annotations
        print('Extracting annotations...')
        extract_tgz('./data/{}'.format(TRAIN_DEVKIT_FILE_NAME), './data/')

    if not os.path.isdir('./data/cars_train/') or not os.listdir('./data/cars_train/'):
        if not os.path.isfile('./data/{}'.format(TRAIN_DATA_FILE_NAME)):
            # Download training dataset
            print('Retrieving training data...')
            download_file(TRAIN_DATA_URL)
        # Extract training images
        print('Extracting images...')
        extract_tgz('./data/{}'.format(TRAIN_DATA_FILE_NAME), './data/')

    if not os.path.isdir('./data/cars_test/') or not os.listdir('./data/cars_test/'):
        if not os.path.isfile('./data/{}'.format(TEST_DATA_FILE_NAME)):
            # Download test dataset
            print('Retrieving test data...')
            download_file(TEST_DATA_URL)
        # Extract test images
        print('Extracting images...')
        extract_tgz('./data/{}'.format(TEST_DATA_FILE_NAME), './data/')

    # Load training annotations and car classes
    train_annos = scipy.io.loadmat('./data/devkit/cars_train_annos.mat')['annotations']
    test_annos = scipy.io.loadmat('./data/devkit/cars_test_annos.mat')['annotations']
    classes = scipy.io.loadmat('./data/devkit/cars_meta.mat')

    train_data = None
    train_labels = None
    train_boxes = None

    # Load/process images in training set
    if not os.path.isfile('./data/train_data.npy') or not os.path.isfile('./data/train_labels.npy') \
            or not os.path.isfile('./data/train_boxes.npy'):
        train_data = np.memmap('./data/train_data.npy', dtype='float32', mode='w+',
                               shape=(len(train_annos[0]), 600, 800, 3))
        train_labels = np.empty((len(train_annos[0]),), dtype='int32')
        train_boxes = np.empty((len(train_annos[0]), 2, 2), dtype='int32')
        i = 0
        for ann in tqdm(train_annos[0], desc='Processing training images', file=sys.stdout, unit='images'):
            img_name = ann[5][0]
            class_no = ann[4][0]

            train_labels[i] = class_no

            im = Image.open('./data/cars_train/{}'.format(img_name))
            im = im.convert('RGB')
            im = resize_image(im)

            train_data[i] = np.array(im)

            # Convert bounding box pixel points to relative points
            p1, p2 = (int(ann[0][0]), int(ann[1][0])), (int(ann[2][0]), int(ann[3][0]))
            box = get_relative_position(p1, p2, im.size)
            train_boxes[i] = box
            i += 1

        np.save('./data/train_labels.npy', train_labels)
        np.save('./data/train_boxes.npy', train_boxes)
    else:
        train_data = np.memmap('./data/train_data.npy', dtype='float32', mode='r',
                               shape=(len(train_annos[0]), 600, 800, 3))
        train_labels = np.load('./data/train_labels.npy')
        train_boxes = np.load('./data/train_boxes.npy')

    test_data = None
    test_boxes = None

    # Load/process images in test set
    if not os.path.isfile('./data/test_data.npy') or not os.path.isfile('./data/test_boxes.npy'):
        test_data = np.memmap('./data/test_data.npy', dtype='float32', mode='w+',
                              shape=(len(test_annos[0]), 600, 800, 3))
        test_boxes = np.empty((len(test_annos[0]), 2, 2), dtype='int32')
        i = 0
        for ann in tqdm(test_annos[0], desc='Processing test images', file=sys.stdout, unit='images'):
            img_name = ann[4][0]

            im = Image.open('./data/cars_test/{}'.format(img_name))
            im = im.convert('RGB')
            im = resize_image(im)

            test_data[i] = np.array(im)

            # Convert bounding box pixel points to relative points
            p1, p2 = (int(ann[0][0]), int(ann[1][0])), (int(ann[2][0]), int(ann[3][0]))
            box = get_relative_position(p1, p2, im.size)
            test_boxes[i] = box
            i += 1

        np.save('./data/test_boxes.npy', test_boxes)
    else:
        test_data = np.memmap('./data/test_data.npy', dtype='float32', mode='r',
                              shape=(len(test_annos[0]), 600, 800, 3))
        test_boxes = np.load('./data/test_boxes.npy')

    # Add processed images to augmentation pipeline and apply operations
    aug = Augmentor.DataPipeline(train_data)
    aug.rotate(0.3, 8, 8)
    aug.skew_left_right(0.4, 0.3)
    aug.flip_left_right(0.5)

    # Save augmented images to './{tmp_dir}/aug'
    # aug.sample(100)
