import pyarrow.parquet as pq
import numpy as np
from header import *

dataset_list = ['CIFAR-10', 'MNIST']


def test_correctness():
    for path in dataset_list:
        for typ in ['train', 'test']:
            logging.info('testing ' + path + '\\' + typ + '_images.pkl')
            images_pkl = unpickle(path + '\\' + typ + '_images.pkl')
            images_parquet = pq.read_table(path + '\\' + typ + '_images.parquet').to_pandas().to_numpy()
            if not np.array_equal(images_parquet, images_pkl):
                logging.error('testing ' + path + '\\' + typ + '_images.pkl failed!')
                return False
            else:
                logging.info('success.')
            logging.info('testing ' + path + '\\' + typ + '_labels.pkl')
            labels_pkl = unpickle(path + '\\' + typ + '_labels.pkl')
            labels_parquet = pq.read_table(path + '\\' + typ + '_labels.parquet').to_pandas().to_numpy()
            if not np.array_equal(labels_parquet, labels_pkl):
                logging.error('testing ' + path + '\\' + typ + '_images.pkl failed!')
                return False
            else:
                logging.info('success!')
    return True


# perf_counter()
logging.info("correctness: {}".format(test_correctness()))



