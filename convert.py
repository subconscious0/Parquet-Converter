import pyarrow as pa
from header import *


def to_parquet(path):
    logging.info('start transforming {} to parquet...'.format(path))
    for typ in ['test', 'train']:
        images = unpickle(path + '\\' + typ + '_images.pkl')
        n, pic_size = images.shape
        table = pa.Table.from_arrays(
            [pa.array(images[:, i]) for i in range(pic_size)],
            schema=pa.schema(
                [pa.field(str(i), pa.uint8()) for i in range(pic_size)]
            )
        )
        toparquet(table, path + '\\' + typ + '_images.parquet')

        labels = unpickle(path + '\\' + typ + '_labels.pkl')
        table = pa.Table.from_arrays(
            [labels.reshape(-1)],
            schema=pa.schema(
                [pa.field('label', pa.uint16())]
            )
        )
        toparquet(table, path + '\\' + typ + '_labels.parquet')

    logging.info('finished transforming {}.'.format(path))


to_parquet('CIFAR-10')
to_parquet('MNIST')
