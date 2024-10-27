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
        print("running backward on {self}" )
        if self._ctx is None:
            return
        
        if self.grad is None and allow_fill:
            # fill in the first grad with one
            assert self.data.szie == 1
            self.grad = np.ones.like(self.data)
    

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
register('dot', Dot)