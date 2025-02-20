#! /usr/bin/env python 
import os 
import unittest
import pdb
import logging
import sys
from tgrad.tensor import Tensor
from tgrad.utils import fetch_mnist, layer_init_uniform
import tgrad.optim as tgrad_optim 
import numpy as np
from tqdm import trange 
# load the mnist dataset


X_train, Y_train, X_test, Y_test = fetch_mnist()
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

#train a model
np.random.seed(1337)
# layer_init = lambda m, h: np.random.uniform(-1, 1, size= (m, h))/np.sqrt(m*h).astype(np.float32)

class TBotNet:
    def __init__(self):
        self.l1 = Tensor(layer_init_uniform(784, 128))
        self.l2 = Tensor(layer_init_uniform(128, 10))

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

class TConvNet:
    def __init__(self):
        conv = 7
        # chans = 4 # characteristics
        chans = 16
        # (batch_size, cin, H, W) => input 
        # (cout, cin, H, W) => self.c1
        self.c1 = Tensor(layer_init_uniform(chans, 1, conv, conv)) # filter
        # (bs, cout, H, W)
        # self.l1 = Tensor(layer_init_uniform(26*26*self.chans, 128)) # serialize
        self.l1 = Tensor(layer_init_uniform(((28-conv+1)**2)*chans, 128)) # serialize
        self.l2 = Tensor(layer_init_uniform(128, 10))


    def forward(self, x: Tensor) -> Tensor:
        x.data = x.data.reshape((-1, 1, 28, 28))
        # x = x.conv2d(self.c1).relu()
        # x = x.reshape(Tensor(np.array((x.shape[0], -1))))

        # 합성곱 레이어 적용
        conv_out = x.conv2d(self.c1)
        # 배치 차원 유지하며 평탄화 (배치사이즈, -1)
        reshaped = conv_out.reshape(Tensor(np.array((conv_out.shape[0], -1))))
        # ReLU 활성화 적용
        x = reshaped.relu()

        # x = x.conv2d(self.c1).reshape(Tensor(np.array((x.shape[0], -1))))
        # x = x.relu()
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

# if os.getenv("CONV") == "1":
#     model = TConvNet()
#     optim = optim.Adam([model.c1, model.l1, model.l2], lr=0.001)
#     step = 400
# else:
#     model = TBotNet()
#     optim = optim.Adam([model.l1, model.l2], lr=0.001)
#     step = 1000

# # original 
# # model = TBotNet()
# # optim = optim.SGD([model.l1, model.l2], 0.001)
# # optim = optim.Adam([model.l1, model.l2], 0.001)

# BS = 128
# losses, accuracies = [], []
# for i in (t := trange(step)):

#     #prepare data
#     samp = np.random.randint(0, X_train.shape[0], size=(BS))
#     x = Tensor(X_train[samp].reshape(-1, 28*28).astype(np.float32))
#     Y = Y_train[samp] # (128, 1)
#     y = np.zeros((len(samp), 10), np.float32) 
#     # y[range(y.shape[0]), Y] = -1.0 # for NLL
#     y[range(y.shape[0]), Y] = -10.0 # correct loss for NLL, torch NLL loss returns one per row
#     y = Tensor(y)

#     # init network
#     out = model.forward(x)

#     #NLL loss funciton 
#     # 간단한 설명:
#     # 1. out는 모델의 예측값으로, 로그 소프트맥스를 통과한 값입니다
#     # 2. y는 실제 레이블을 -1.0으로 인코딩한 값입니다
#     # 3. mul()로 두 텐서를 곱하고 mean()으로 평균을 구해 최종 손실값을 계산합니다
#     # 실제 예시:
#     # 손실 계산 과정
#     # 1. out * y 계산
#     # [[-2.3*0, -1.5*0, -0.1*(-1)],
#     #  [-1.8*(-1), -2.1*0, -0.4*0]]
#     # = [[0, 0, 0.1],
#     #    [1.8, 0, 0]]

#     # NLL loss function for training 
#     # 2. 평균 계산
#     # (0 + 0 + 0.1 + 1.8 + 0 + 0) / 6 = 0.317
#     loss = out.mul(y).mean()
#     loss.backward()

#     # evaluation
#     cat = np.argmax(out.data, axis=1)
#     accuracy = (cat == Y).mean()

#     # SGD
#     # print(model.l1.grad)
#     # lr = 0.001
#     # model.l1.data = model.l1.data - lr*model.l1.grad
#     # model.l2.data = model.l2.data - lr*model.l2.grad
#     optim.step()
    

#     # printing
#     loss = loss.data
#     losses.append(loss)
#     accuracies.append(accuracy)
#     # t.set_description(f"loss: {losses}, accuracy: {accuracies}")

class TestMNIST(unittest.TestCase):
    def test_mnist(self):
        # 로깅 설정
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

        # if os.getenv("TRACE") == "1":
        #     sys.settrace(trace_calls)

        # if os.getenv("DEBUG") == "1":
        #     pdb.set_trace()  # 디버깅 포인트 설정
        if os.getenv("CONV") == "1":
            model = TConvNet()
            optim = tgrad_optim.Adam([model.c1, model.l1, model.l2], lr=0.001)
            steps = 400
        else:
            model = TBotNet()
            # optim = tgrad_optim.Adam([model.l1, model.l2], lr=0.001)
            optim = tgrad_optim.RMSprop([model.l1, model.l2], lr=0.001)
            steps = 1000

        # original 
        # model = TBotNet()
        # optim = optim.SGD([model.l1, model.l2], 0.001)
        # optim = optim.Adam([model.l1, model.l2], 0.001)

        BS = 128
        losses, accuracies = [], []
        for i in (t := trange(steps)):
            #prepare data
            samp = np.random.randint(0, X_train.shape[0], size=(BS))
            x = Tensor(X_train[samp].reshape(-1, 28*28).astype(np.float32))
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

            # 학습 과정 로깅
            if i % 100 == 0:
                logger.debug(f"Step {i}: Loss = {float(loss):.4f}, Accuracy = {float(accuracy):.4f}")
                logger.debug(f"Gradients - l1: {float(model.l1.grad.mean()):.4f}, l2: {float(model.l2.grad.mean()):.4f}")
        
        def numpy_eval():
            # Y_test_preds_out = model.forward(Tensor(X_test.reshape(-1, 28*28)))
            Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32)))
            Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
            return (Y_test_preds == Y_test).mean()

        accuracy = numpy_eval()
        print(f"test set accuracy is {accuracy}")
        assert accuracy > 0.95

if __name__ == '__main__':
    # for test beginner
    # print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # model = TBotNet()
    # optim = optim.Adam([model.l1, model.l2], 0.001)


    # original 
    unittest.main()
