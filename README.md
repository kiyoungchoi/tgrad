tinygrad to tgrad 

### how to get started ( with setup.py )
```
pip install -e .
```

### get started 
```python
python test/test_mnist.py
CONV=1 python test/test_mnist.py
```

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

### 학습 진행사항


## 📑 [학습 진행사항 완료 리스트](https://github.com/tinygrad/tinygrad/commits/master/?since=2020-10-19&until=2020-10-30&after=910ae260cd1d45a1326299081e6cc70832cfd21f+69) 
- [extracting jacobian and test_jacobian](https://github.com/tinygrad/tinygrad/commit/1561d3b9c01675f70a37e2b39674465158fb5abd)