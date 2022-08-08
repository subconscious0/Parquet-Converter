import random
import time
from header import *

dataset_list = ['CIFAR-10', 'MNIST']


def test_pickle():
    ret = []
    for path in dataset_list:
        for typ in ['train', 'test']:
            ret.append(unpickle(path + '\\' + typ + '_images.pkl'))
            ret.append(unpickle(path + '\\' + typ + '_labels.pkl'))
    return ret


def test_parquet():
    ret = []
    for path in dataset_list:
        for typ in ['train', 'test']:
            ret.append(unparquet(path + '\\' + typ + '_images.parquet'))
            ret.append(unparquet(path + '\\' + typ + '_labels.parquet'))
    return ret


def test_parquet_partial_read():
    ret = []
    for path in dataset_list:
        for typ in ['train', 'test']:
            pf = pq.ParquetFile(os.path.join(workingDirectory, path + '\\' + typ + '_images.parquet'))
            ret.append(pf.read_row_group(random.choice(list(range(pf.num_row_groups)))))
            pf = pq.ParquetFile(os.path.join(workingDirectory, path + '\\' + typ + '_labels.parquet'))
            ret.append(pf.read_row_group(random.choice(list(range(pf.num_row_groups)))))
    return ret


# perf_counter()
start_time = time.perf_counter()
test_parquet()
logging.info("parquet time: {}".format(time.perf_counter() - start_time))

start_time = time.perf_counter()
test_parquet_partial_read()
logging.info("parquet partial time: {}".format(time.perf_counter() - start_time))

start_time = time.perf_counter()
test_pickle()
logging.info("pickle time: {}".format(time.perf_counter() - start_time))

