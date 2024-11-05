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

# # test
# import torch

# class MyReLU(torch.autograd.Function):
#   @staticmethod
#   def forward(ctx, input):
#     ctx.save_for_backward(input)
#     return input.clamp(min=0)

#   @staticmethod
#   def backward(ctx, grad_output):
#     print(ctx)
#     print(f"saved_tensors: {ctx.saved_tensors}")
#     input, = ctx.saved_tensors
#     grad_input = grad_output.clone()
#     grad_input[input < 0] = 0
#     return grad_input

# t = [
#   torch.tensor([-1., 1], requires_grad=True),
#   torch.tensor([-2., 2], requires_grad=True)
# ] 
# for i in t:
#   tr = MyReLU.apply(i)
#   tr.mean().backward()
#   print(tr, i.grad)
# exit(0)


# original 
model = TBotNet()

lr = 0.001
BS = 128
losses, accuracies = [], []
for i in (t := trange(10)):

    #prepare data
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    x = Tensor(X_train[samp].reshape(-1, 28*28))
    Y = Y_train[samp] # (128, 1)
    y = np.zeros((len(samp), 10), np.float32) 
    y[range(y.shape[0]), Y] = -1.0 # for NLL
    y = Tensor(y)

    # init network
    outs = model.forward(x)

    #NLL loss funciton 
    # 간단한 설명:
    # 1. outs는 모델의 예측값으로, 로그 소프트맥스를 통과한 값입니다
    # 2. y는 실제 레이블을 -1.0으로 인코딩한 값입니다
    # 3. mul()로 두 텐서를 곱하고 mean()으로 평균을 구해 최종 손실값을 계산합니다
    # 실제 예시:
    # 손실 계산 과정
    # 1. outs * y 계산
    # [[-2.3*0, -1.5*0, -0.1*(-1)],
    #  [-1.8*(-1), -2.1*0, -0.4*0]]
    # = [[0, 0, 0.1],
    #    [1.8, 0, 0]]

    # NLL loss function for training 
    # 2. 평균 계산
    # (0 + 0 + 0.1 + 1.8 + 0 + 0) / 6 = 0.317
    loss = outs.mul(y).mean()
    loss.backward()

    # evaluation
    cat = np.argmax(outs.data, axis=1)
    accuracy = (cat == Y).mean()

    # SGD
    # print(model.l1.grad)
    model.l1.data = model.l1.data - lr*model.l1.grad
    model.l2.data = model.l2.data - lr*model.l2.grad

    # printing
    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    # t.set_description(f"loss: {losses}, accuracy: {accuracies}")

def numpy_eval():
  Y_test_preds_out = model.forward(Tensor(X_test.reshape(-1, 28*28)))
  Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
  return (Y_test_preds == Y_test).mean()


accuracy = numpy_eval()
print(f"test set accuracy is {accuracy}")
assert accuracy > 0.95
