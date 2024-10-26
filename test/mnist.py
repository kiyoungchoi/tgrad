#! /usr/bin/env python 
from tgrad.tensor import Tensor
import numpy as np
from tqdm import trange 
# load the mnist dataset

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
  return numpy.frombuffer(gzip.decompress(dat), dtype='uint8').copy()

X_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz")[8:]

#train a model

def layer_init(m, h):
    ret = np.random.uniform(-1, 1, size= (m, h))/np.sqrt(m*h)
    return ret.astype(np.float32)

class TBotNet:
    def __init__(self):
        self.l1 = Tensor(layer_init(784, 128))
        self.l2 = Tensor(layer_init(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TBotNet()

lr = 0.01
BS = 128
losses, accuracies = [], []
for i in (t := trange(1000)):

    #prepare data
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    x = Tensor(X_train[samp].reshape(-1, 28*28))
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32) 
    y[range(y.shape[0]), Y] = -1.0
    y = Tensor(y)

    # init network
    outs = model.forward(x)

    #NLL loss funciton 
    # loss = outs.mul(y).mean()
    # print(loss)
    # loss.backward()
    

