from functools import partialmethod
import numpy as np


# *** start with three base classes ***

class Context:
    def __init__(self, arg, *tensors):
        self.arg = arg
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

class Tensor:
    def __init__(self, data):
        # print(data.shape, data)
        if type(data) != np.ndarray:
            print(f"error constructing tensor with {data} ")
            assert(False)
        self.data = data
        self.grad = None
    
        # internal variables used for autograd graph construnction 
        self._ctx = None
    
    def __str__(self):
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

        print("running backward on", self)
        if self._ctx is None:
            return
        
        if self.grad is None and allow_fill:
            # fill in the first grad with one
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)
        
        assert(self.grad is not None)

        grads = self._ctx.arg.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print("grad shape must match tensor shape in {se.f_ctx.arg}, {g.shape} != {t.data.shape}")
                assert(False)
            t.grad = g
            t.backward(False)
    
    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div)
    

# class Function:
#     def apply(self, arg, *x):
#         ctx = Context(arg, self, *x)
#         ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
#         ret._ctx = ctx
#         return ret 

# def register(name, fxn):
#     setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# class Dot(Function):
#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.save_for_backward(input, weight)
#         return input.dot(weight)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight = ctx.saved_tensors
#         grad_input = grad_output.dot(weight.T)
#         grad_weight = grad_output.T.dot(input).T
#         return grad_input, grad_weight
# register('dot',Dot)
class Function:
    def apply(self, arg, *x):
        ctx = Context(arg, self, *x)
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
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
        # it is not layer, just act for input 
        ctx.save_for_backward(input)
        return np.maximum(input, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
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