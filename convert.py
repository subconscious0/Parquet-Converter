import pyarrow.parquet as pq
import pyarrow as pa
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def to_parquet(path):
    print('start transforming ' + path + ' to parquet')
    for typ in ['test', 'train']:
        images = unpickle(path + '/' + typ + '_images.pkl')
        n, pic_size = images.shape
        table = pa.Table.from_arrays(
            [pa.array(images[:, i]) for i in range(pic_size)],
            schema=pa.schema(
                [pa.field(str(i), pa.uint8()) for i in range(pic_size)]
            )
        )
        pq.write_table(table, path + '/' + typ + '_images.parquet')

        labels = unpickle(path + '/' + typ + '_labels.pkl')
        table = pa.Table.from_arrays(
            [labels],
            schema=pa.schema(
                [pa.field('label', pa.uint16())]
            )
        )
        pq.write_table(table, path + '/' + typ + '_labels.parquet')

    print('finished transforming' + path)


to_parquet('CIFAR-10')
to_parquet('MNIST')
