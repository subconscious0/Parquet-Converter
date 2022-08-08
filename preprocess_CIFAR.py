import numpy as np
from header import *


def process_cifar_test():
    logging.info('start processing test...')
    test = unpickle('CIFAR-10\\test_batch')

    images = test[b'data']
    labels = np.array(test[b'labels'])[np.newaxis].T

    topickle(images, 'CIFAR-10\\test_images.pkl')
    topickle(labels, 'CIFAR-10\\test_labels.pkl')
    logging.info('test processed')


def process_cifar_train():
    logging.info('start processing train...')
    image_list = []
    label_list = []

    for i in range(1, 6):
        train_batch = unpickle('CIFAR-10\\data_batch_' + str(i))
        image_list.append(train_batch[b'data'])
        label_list.append(np.array(train_batch[b'labels']))

    images = np.concatenate(tuple(image_list), axis=0)
    labels = np.concatenate(tuple(label_list), axis=0)[np.newaxis].T

    topickle(images, 'CIFAR-10\\train_images.pkl')
    topickle(labels, 'CIFAR-10\\train_labels.pkl')
    logging.info('train processed.')


process_cifar_train()
process_cifar_test()




