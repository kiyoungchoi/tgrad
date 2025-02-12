import numpy as np
from tgrad.tensor import Tensor
from typing import List

class Optimizer:
  def __init__(self, params: List[Tensor]):
    self.params = params


class SGD(Optimizer):
  def __init__(self, params: List[Tensor], lr: float = 0.001):
    super(SGD, self).__init__(params)
    self.lr = lr

  def step(self):
    for t in self.params:
      t.data -= self.lr * t.grad
    

class Adam(Optimizer):
  def __init__(self, params: List[Tensor], lr: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    super(Adam, self).__init__(params)
    self.lr = lr
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.t = 0

    self.m = [np.zeros_like(t.data) for t in self.params]
    self.v = [np.zeros_like(t.data) for t in self.params]

  def step(self):
    for i, t in enumerate(self.params):
      self.t +=  1
      self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
      self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad)
      # more Readability in tgrad vs more fast in pytorch
      mhat = self.m[i] / (1. - self.b1**self.t)
      vhat = self.v[i] / (1. - self.b2**self.t)
      t.data -= (self.lr * mhat)/(np.sqrt(vhat) + self.eps)


class RMSprop(Optimizer):
  def __init__(self, params: List[Tensor], lr: float = 0.001, decay: float = 0.9, eps: float = 1e-6):
    super(RMSprop, self).__init__(params)
    self.lr: float = lr
    self.decay: float = decay
    self.eps: float = eps
    self.t: int = 0
    self.v: List[np.ndarray] = [np.zeros_like(t.data) for t in self.params]
  def step(self):
    self.t += 1
    for i, t in enumerate(self.params):
      self.v[i] = self.decay * self.v[i] + (1 - self.decay) * np.square(t.grad)
      t.data -= (self.lr / np.sqrt(self.v[i] + self.eps)) * t.grad