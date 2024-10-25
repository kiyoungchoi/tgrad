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
        print(data.shape, data)