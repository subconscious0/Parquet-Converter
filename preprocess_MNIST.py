import numpy as np
from header import *


def process_mnist():
    logging.info('start processing mnist...')
    data = unpickle('MNIST\\mnist.pkl')
    topickle(data['training_images'], 'MNIST\\train_images.pkl')
    topickle(data['test_images'], 'MNIST\\test_images.pkl')
    topickle(data['training_labels'][np.newaxis].T, 'MNIST\\train_labels.pkl')
    topickle(data['test_labels'][np.newaxis].T, 'MNIST\\test_labels.pkl')

    logging.info('mnist processed.')


process_mnist()




