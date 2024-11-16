#! /usr/bin/env python 
from tgrad.tensor import Tensor
from tgrad.utils import fetch_mnist, layer_init_uniform
import tgrad.optim as optim
import numpy as np
from tqdm import trange 
# load the mnist dataset


X_train, Y_train, X_test, Y_test = fetch_mnist()

#train a model
np.random.seed(1337)
# layer_init = lambda m, h: np.random.uniform(-1, 1, size= (m, h))/np.sqrt(m*h).astype(np.float32)

class TBotNet:
    def __init__(self):
        self.l1 = Tensor(layer_init_uniform(784, 128))
        self.l2 = Tensor(layer_init_uniform(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

class TConvNet:
    def __init__(self):
        self.chans = 4 # characteristics
        # (batch_size, cin, H, W) => input 
        # (cout, cin, H, W) => self.c1
        self.c1 = Tensor(layer_init_uniform(self.chans, 1, 3, 3)) # filter
        # (bs, cout, H, W)
        self.l1 = Tensor(layer_init_uniform(26*26*self.chans, 128)) # serialize
        self.l2 = Tensor(layer_init_uniform(128, 10))


    def forward(self, x):
        x.data = x.data.reshape((-1, 1, 28, 28))
        x = x.conv2d(self.c1).reshape(Tensor(np.array(-1, 26*26*self.chans))).relu()
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
optim = optim.SGD([model.l1, model.l2], 0.001)
# optim = optim.Adam([model.l1, model.l2], 0.001)

BS = 128
losses, accuracies = [], []
for i in (t := trange(200)):

    #prepare data
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    x = Tensor(X_train[samp].reshape(-1, 28*28))
    Y = Y_train[samp] # (128, 1)
    y = np.zeros((len(samp), 10), np.float32) 
    # y[range(y.shape[0]), Y] = -1.0 # for NLL
    y[range(y.shape[0]), Y] = -10.0 # correct loss for NLL, torch NLL loss returns one per row
    y = Tensor(y)

    # init network
    out = model.forward(x)

    #NLL loss funciton 
    # 간단한 설명:
    # 1. out는 모델의 예측값으로, 로그 소프트맥스를 통과한 값입니다
    # 2. y는 실제 레이블을 -1.0으로 인코딩한 값입니다
    # 3. mul()로 두 텐서를 곱하고 mean()으로 평균을 구해 최종 손실값을 계산합니다
    # 실제 예시:
    # 손실 계산 과정
    # 1. out * y 계산
    # [[-2.3*0, -1.5*0, -0.1*(-1)],
    #  [-1.8*(-1), -2.1*0, -0.4*0]]
    # = [[0, 0, 0.1],
    #    [1.8, 0, 0]]

    # NLL loss function for training 
    # 2. 평균 계산
    # (0 + 0 + 0.1 + 1.8 + 0 + 0) / 6 = 0.317
    loss = out.mul(y).mean()
    loss.backward()

    # evaluation
    cat = np.argmax(out.data, axis=1)
    accuracy = (cat == Y).mean()

    # SGD
    # print(model.l1.grad)
    # lr = 0.001
    # model.l1.data = model.l1.data - lr*model.l1.grad
    # model.l2.data = model.l2.data - lr*model.l2.grad
    optim.step()
    

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
