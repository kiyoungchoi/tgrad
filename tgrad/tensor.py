from functools import partialmethod
import numpy as np


# *** start with three base classes ***

class Context:
    def __init__(self, arg, *tensor):
        self.arg = arg
        self.parents = tensors
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        self.saved_tensors_extend(x)

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
    
