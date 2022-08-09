# Parquet-Converter
A Parquet Converter For Open-Source ML Datasets

### Prerequisite

Numpy: [install](https://numpy.org/install/)

Pyarrow: [install](https://arrow.apache.org/docs/python/install.html)

### Usage - download CIFAR-10

Download the Python version of data from [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and put it under `./CIFAR-10`.

### Usage - download MNIST

Download and parse data using [this](https://github.com/hsjeong5/MNIST-for-Numpy) and put it under `./MNIST`.

### Usage - Preprocess and Convert

Run the following command:

```
python preprocess_CIFAR.py
python preprocess_MNIST.py
python convert.py
```

These programs create pickle files and parquet files under `./CIFAR-10` and `./MNIST`.

### Usage - testing and benchmarking

To check the correctness of the created files, run `python test.py`.

To benchmark the performance, run `python benchmark.py`.
