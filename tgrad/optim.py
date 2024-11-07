class SGD:
  def __init__(self, tensors, lr=0.001):
    self.tensors = tensors
    self.lr = lr

  def step(self):
    for t in self.tensors:
      t.data -= self.lr * t.grad
    

