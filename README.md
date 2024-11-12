tinygrad to tgrad 

understanding the process 


### You can even train neural networks with tinygrad (from /test/mnist.py)

```python
from tgrad.tensor import Tensor
import tgrad.optim as optim
from tgrad.untils import layer_init_uniform

class TBotNet:
    def __init__(self):
        self.l1 = Tensor(layer_init_uniform(784, 128))
        self.l2 = Tensor(layer_init_uniform(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TBotNet()
optim = optim.SGD([model.l1, model.l2], lr=0.001)

# ... and complete like pytorch, with (x, y) data
out = model.forward(x)
loss = out.mul(y).mean()
loss.backward()
optim.step() 

```