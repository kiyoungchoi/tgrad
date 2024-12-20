import numpy as np
import torch
import unittest
from tgrad.tensor import Tensor, Conv2D
from tgrad.gradcheck import numerical_jacobian, gradcheck

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)


class TestTgrad(unittest.TestCase):
    def test_backward_pass(self):
        def test_tgrad():
            x = Tensor(x_init)
            W = Tensor(W_init)
            m = Tensor(m_init)
            out = x.dot(W)
            out = out.relu()
            out = out.logsoftmax()
            out = out.mul(m)
            out = out.add(m)
            out = out.sum()
            out.backward()
            return out.data, x.grad, W.grad
            
        def test_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            m = torch.tensor(m_init)
            out = x.matmul(W)
            out = out.relu()
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, W.grad

        for x, y in zip(test_tgrad(), test_pytorch()):
            print(x, y)
            np.testing.assert_allclose(x, y, atol=1e-6)

    def test_jacobian(self):
        W = np.random.RandomState(1337).random((10, 5))
        x = np.random.RandomState(7331).random((1, 10)) - 0.5

        torch_x = torch.tensor(x, requires_grad=True)
        torch_W = torch.tensor(W, requires_grad=True)
        torch_func = lambda x: torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)
        torch_out = torch_func(torch_x)

        # autograd.grad computes the _sum_ of gradients of given tensors
        J_sum = torch.autograd.grad(list(torch_out[0]), torch_x)[0].squeeze().numpy()

        t_x = Tensor(x)
        t_W = Tensor(W)
        t_func = lambda x: x.dot(t_W).relu().logsoftmax()
        NJ = numerical_jacobian(t_func, t_x)
        NJ_sum = NJ.sum(axis=-1)

        np.testing.assert_allclose(J_sum, NJ_sum, atol=1e-5)

    
    def test_gradcheck(self):
        class TgradModel:
            def __init__(self, weight_init):
                self.l1 = Tensor(weight_init)
            def forward(self, x):
                return x.dot(self.l1).relu().logsoftmax()
        
        class TorchModel(torch.nn.Module):
            def __init__(self, weights_init):
                super(TorchModel, self).__init__()
                self.l1 = torch.nn.Linear(*weights_init.shape, bias = False)
                self.l1.weight =torch.nn.Parameter(torch.tensor(weights_init.T, requires_grad=True))
            def forward(self, x):
                return torch.nn.functional.log_softmax(self.l1(x).relu(), dim=1)
        
        layer_weights = np.random.RandomState(1337).random((10, 5))
        input_data = np.random.RandomState(7331).random((1,10)) - 0.5

        torch_input = torch.tensor(input_data, requires_grad = True)
        torch_model = TorchModel(layer_weights)
        # torch_out = torch_model(torch_input)
        # # autograd.grad computes the _sum_ of gradients of given tensors
        # J_sum = torch.autograd.grad(list(torch_out[0]), torch_input)[0].squeeze().numpy()

        tgrad_model = TgradModel(layer_weights)
        tgrad_input = Tensor(input_data)
        # tgrad_out = tgrad_model.forward(tgrad_input)

        # NJ = numerical_jacobian(tgrad_model, tgrad_input)
        # NJ_sum = NJ.sum(axis = -1)

        # # checking the numerical approx. of J is close to the one provided autograd
        # np.testing.assert_allclose(J_sum, NJ_sum, atol = 1e-5)
        # test gradcheck
        gradcheck_test, _, _ = gradcheck(tgrad_model.forward, tgrad_input)
        self.assertTrue(gradcheck_test)
        # coarse approx. since a "big" eps and the non-linearities of the model
        gradcheck_test, j, nj = gradcheck(tgrad_model.forward, tgrad_input, eps = 0.1)
        self.assertFalse(gradcheck_test)


    
    def test_conv2d(self):
        x = torch.randn((5,2,10,7), requires_grad=True)
        w = torch.randn((4,2,3,3), requires_grad=True)
        xt = Tensor(x.detach().numpy())
        wt = Tensor(w.detach().numpy())

        out = torch.nn.functional.conv2d(x,w)
        ret = Conv2D.apply(Conv2D, xt, wt)
        np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=1e-5)

        out.mean().backward()
        ret.mean().backward()
        
        np.testing.assert_allclose(w.grad, wt.grad, atol=1e-5)
        np.testing.assert_allclose(x.grad, xt.grad, atol=1e-5)

if __name__ == '__main__':
    unittest.main()

