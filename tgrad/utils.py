import numpy as np

def fetch_mnist():
    def fetch(url):
        import requests, gzip, os, hashlib, numpy
        fp = os.path.join('/tmp', hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, 'rb') as f:
                dat = f.read()
        else:
            with open(fp, 'wb') as f:
                dat = requests.get(url).content
                f.write(dat)
        return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
    X_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train , X_test, Y_test