import pyarrow.parquet as pq
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


def unparquet(file):
    return pq.read_table(os.path.join(workingDirectory, file))


def toparquet(data, file):
    pq.write_table(data,
                   os.path.join(workingDirectory, file),
                   row_group_size=8 * 1024,
                   compression='GZIP'
                   )


