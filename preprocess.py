import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def topickle(data, file):
    with open(file, 'wb') as fo:
        pickle.dump(data, fo)

def process_cifar_test():
    print('start processing test...')
    test = unpickle('CIFAR-10/test_batch')

    images = test[b'data']
    labels = np.array(test[b'labels'])
    print(images.shape, labels.shape)

    topickle(images, 'CIFAR-10/test_images.pkl')
    topickle(labels, 'CIFAR-10/test_labels.pkl')
    print('test processed')

def process_cifar_train():
    print('start processing train...')
    image_list = []
    label_list = []

    for i in range(1, 6):
        train_batch = unpickle('CIFAR-10/data_batch_' + str(i))
        image_list.append(train_batch[b'data'])
        label_list.append(np.array(train_batch[b'labels']))

    images = np.concatenate(tuple(image_list), axis=0)
    labels = np.concatenate(tuple(label_list), axis=0)
    print(images.shape, labels.shape)

    topickle(images, 'CIFAR-10/train_images.pkl')
    topickle(labels, 'CIFAR-10/train_labels.pkl')
    print('train processed')

def process_mnist():
    print('start processing mnist...')
    data = unpickle('MNIST/mnist.pkl')
    print(data.keys())
    topickle(data['training_images'], 'MNIST/train_images.pkl')
    topickle(data['test_images'], 'MNIST/test_images.pkl')
    topickle(data['training_labels'], 'MNIST/train_labels.pkl')
    topickle(data['test_labels'], 'MNIST/test_labels.pkl')

    print("mnist processed")


process_cifar_train()
process_cifar_test()
process_mnist()




