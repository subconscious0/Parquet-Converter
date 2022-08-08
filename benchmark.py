import pyarrow.parquet as pq
import time
from header import *

dataset_list = ['CIFAR-10', 'MNIST']


def test_pickle():
    ret = []
    for path in dataset_list:
        for typ in ['train', 'test']:
            temp = unpickle(path + '\\' + typ + '_images.pkl')
            ret.append(temp)
            temp = unpickle(path + '\\' + typ + '_labels.pkl')
            ret.append(temp)
    return ret


def test_parquet():
    ret = []
    for path in dataset_list:
        for typ in ['train', 'test']:
            temp = pq.read_table(path + '\\' + typ + '_images.parquet')
            ret.append(temp)
            temp = pq.read_table(path + '\\' + typ + '_labels.parquet')
            ret.append(temp)
    return ret


# perf_counter()
start_time = time.perf_counter()
test_parquet()
logging.info("parquet time: {}".format(time.perf_counter() - start_time))

start_time = time.perf_counter()
test_pickle()
logging.info("pickle time: {}".format(time.perf_counter() - start_time))


