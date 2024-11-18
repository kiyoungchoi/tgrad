from functools import partialmethod
import numpy as np


# *** start with two base classes ***
class Tensor:
    def __init__(self, data):
        # print(data.shape, data)
        if type(data) != np.ndarray:
            print(f"error constructing tensor with {data} ")
            assert(False)
        if data.dtype == np.float64:
            print(f"are you sure you want float64 in {data}?")
        self.data = data
        self.grad = None
    
        # internal variables used for autograd graph construnction 
        self._ctx = None
    
    def __repr__(self):
        return f"Tensor {self.data} with grad {self.grad}"
    
    def backward(self, allow_fill=True):
        # # 텐서 생성 example
        # a = Tensor(np.array([2.0]))
        # b = Tensor(np.array([3.0]))

        # # 연산 (덧셈)
        # c = a + b  # 내부적으로 add 함수가 호출됨

        # # 역전파 실행
        # c.backward()

        # print(a.grad)  # 출력: 1.0
        # print(b.grad)  # 출력: 1.0

        # print("running backward on", self)
        if self._ctx is None:
            return
        
        if self.grad is None and allow_fill:
            # fill in the first grad with one
            # this is "implicit gradient creation"
            
            assert self.data.size == 1
            # 순전파 결과:
            # [ 0.73245234 -1.52361438]

            # 그래디언트 초기화:
            # [[1. 1.]]
            self.grad = np.ones_like(self.data)
        
        assert(self.grad is not None)

        grads = self._ctx.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g is None:
                continue
            if g.shape != t.data.shape:
                print("grad shape must match tensor shape in {se.f_ctx}, {g.shape} != {t.data.shape}")
                assert(False)
            t.grad = g
            t.backward(False)
    
    def mean(self):
        div = Tensor(np.array([1/self.data.size], dtype=self.data.dtype))
        return self.sum().mul(div)
    
# An instaniation of the Function is the Context
class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        # # extend() 사용 예제
        # list1 = [1, 2, 3]
        # list2 = [4, 5, 6]

        # # extend() 사용
        # list1.extend(list2)
        # print(list1)  # 출력: [1, 2, 3, 4, 5, 6]
        self.saved_tensors.extend(x)

    # note that due to how partialmethod works, self and arg are switched
    def apply(self, arg, *x):
        # ctx = arg(self, *x)
        # ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        # ret._ctx = ctx
        # return ret
        # print(type(arg) == Tensor) # to test
        if type(arg) == Tensor:
            op = self
            x = [arg]+list(x)
        else: 
            op = arg
            x = [self] + list(x)
        ctx = op(*x)
        ret = Tensor(op.forward(ctx, *[t.data for t in x]))
        ret._ctx = ctx
        return ret

def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        # input is with a lot of 0's
        # weight is with random.uniform()
        ctx.save_for_backward(input, weight)
        return input.dot(weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input, grad_weight
register('dot', Dot)

class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        # 순전파: 입력값을 저장하고 ReLU 연산 수행
        # it is not layer, just act for input 
        ctx.save_for_backward(input)
        return np.maximum(input, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 역전파: 저장된 입력값을 사용하여 기울기 계산
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
register('relu',ReLU)

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsumexp(x):
            c = x.max(axis=1)
            # 예시: 입력값이 [300, 200, 500]일 때
            # 1. c = 500 (최대값)
            # 2. x-c: [300-500, 200-500, 500-500] = [-200, -300, 0]
            # 3. 이렇게 하는 이유는 지수 함수의 수치적 안정성을 위해서입니다
            # 4. exp(-200) + exp(-300) + exp(0) 는 exp(300) + exp(200) + exp(500) 보다
            #    오버플로우 위험이 훨씬 적습니다
            return c + np.log(np.exp(x-c.reshape(-1,1)).sum(axis=1))
        output = input - logsumexp(input).reshape(-1, 1)
        ctx.save_for_backward(output)
        return output 
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape(-1, 1)

register('logsoftmax', LogSoftmax)

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x*y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y*grad_output, x*grad_output

register('mul', Mul)

class Sum(Function):
    # a = Tensor(np.array([2.0]))
    # b = Tensor(np.array([3.0]))
    # c = a.add(b)  # 덧셈 연산
    # c.backward()

    # print(a.grad)  # 출력: 1.0
    # print(b.grad)  # 출력: 1.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)
register('sum', Sum)


class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x+y

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output, grad_output
register('add', Add)

from numba import jit, float32

@jit(nopython=True)
def conv2d_inner_forward(x, w):
    cout, cin, H, W = w.shape
    ret = np.zeros((x.shape[0], cout, x.shape[2]-(H-1), x.shape[3]-(W-1)), dtype=w.dtype)
    for j in range(H):
        for i in range(W):
            tw = w[:, :, j, i] # 차원 축소됨, (1, 2)
            for Y in range(ret.shape[2]):
                for X in range(ret.shape[3]):
                    ret[:, :, Y, X] += x[:, :, Y+j, X+i].dot(tw.T)
    return ret

@jit(nopython=True)
def conv2d_inner_backward(grad_output, x, w):
    dx = np.zeros_like(x) # format 
    dw = np.zeros_like(w) # format
    cout, cin, H, W = w.shape  #filter
    for j in range(H):
        for i in range(W):
            tw = w[:, :, j, i]
            for Y in range(grad_output.shape[2]):
                for X in range(grad_output.shape[3]):
                    gg = grad_output[:, :, Y, X]
                    tx = x[:, :, Y, X]
                    dx[:, :, Y+j, X+j] += gg.dot(tw)
                    dw[:, :, j, i] += gg.T.dot(tx)
    return dx, dw

class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        # cout, cin, H, W = w.shape
        # ret = np.zeros((x.shape[0], cout, x.shape[2]-(H-1), x.shape[3]-(W-1)), dtype=w.dtype)
        # for j in range(H):
        #     for i in range(W):
        #         tw = w[:, :, j, i]
        #         for Y in range(ret.shape[2]):
        #             for X in range(ret.shape[3]):
        #                 ret[:, :, Y, X] += x[:, :, Y+j, X+i].dot(tw.T)
        # return ret
        return conv2d_inner_forward(x, w)
    
    @staticmethod
    def backward(ctx, grad_output):
        # x, w = ctx.saved_tensors
        # dx = np.zeros_like(x)
        # dw = np.zeros_like(w)
        # cout, cin, H, W = w.shape
        # for j in range(H):
        #     for i in range(W):
        #         tw = w[:, :, j, i]
        #         for Y in range(grad_output.shape[2]):
        #             for X in range(grad_output.shape[3]):
        #                 gg = grad_output[:, :, Y, X]
        #                 tx = x[:, :, Y+j, X+i]
        #                 dx[:, :, Y+j, X+i] += gg.dot(tw)
        #                 dw[:, :, j, i] += gg.T.dot(tx)
        # return dx, dw
        return conv2d_inner_backward(grad_output, *ctx.saved_tensors)
register('conv2d', Conv2D)

class Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        in_shape,  = ctx.saved_tensors
        return grad_output.reshape(in_shape), None 
register('reshape', Reshape)