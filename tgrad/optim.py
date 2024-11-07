class Optimizer:
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr


class SGD(Optimizer):
  def step(self):
    for t in self.params:
      t.data -= self.lr * t.grad
    

class Adam(Optimizer):
  def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
    super(Adam, self).__init__(params, lr)
    self.b1 = b1
    self.b2 = b2
    self.eps = eps

  def step(self):
    pass
