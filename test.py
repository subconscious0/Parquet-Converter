import pyarrow.parquet as pq
import numpy as np
from header import *

dataset_list = ['CIFAR-10', 'MNIST']


def test_correctness(path, typ):
    logging.info('testing ' + path + '\\' + typ + '_images.pkl')
    images_pkl = unpickle(path + '\\' + typ + '_images.pkl')
    images_parquet = pq.read_table(path + '\\' + typ + '_images.parquet').to_pandas().to_numpy()
    assert np.array_equal(images_parquet, images_pkl)
    logging.info('success.')
    logging.info('testing ' + path + '\\' + typ + '_labels.pkl')
    labels_pkl = unpickle(path + '\\' + typ + '_labels.pkl')
    labels_parquet = pq.read_table(path + '\\' + typ + '_labels.parquet').to_pandas().to_numpy()
    assert np.array_equal(labels_parquet, labels_pkl)
    logging.info('success.')


for path in dataset_list:
    for typ in ['train', 'test']:
        test_correctness(path, typ)
logging.info('passed.')



