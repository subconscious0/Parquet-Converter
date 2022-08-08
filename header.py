import logging
import pickle
import sys
import os

logging.basicConfig(level=logging.DEBUG)
workingDirectory = os.path.realpath(sys.argv[0]) + '\\..\\'


def unpickle(file):
    with open(os.path.join(workingDirectory, file), 'rb') as fo:
        arr = pickle.load(fo, encoding='bytes')
    return arr


def topickle(data, file):
    with open(os.path.join(workingDirectory, file), 'wb') as fo:
        pickle.dump(data, fo)
