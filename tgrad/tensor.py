from typing import Union, Type, List, Tuple, Callable, Any, Optional
from functools import partialmethod
import numpy as np


# *** start with two base classes ***
class Tensor:
    def __init__(self, data: np.ndarray) -> None:
        # print(data.shape, data)
        if not isinstance(data, np.ndarray):
            print(f"error constructing tensor with {data} ")
            assert False
        if data.dtype == np.float64:
            # print(f"are you sure you want float64 in {data}?")
            pass
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
    
        # internal variables used for autograd graph construnction 
        self._ctx: Optional["Function"] = None
    
    def __repr__(self) -> str:
        return f"Tensor {self.data} with grad {self.grad}"
    
    def backward(self, allow_fill: bool = True) -> None:
        # # 텐서 생성 example
        # a = Tensor(np.array([2.0]))
        # b = Tensor(np.array([3.0]))

        # # 연산 (덧셈)
        # c = a + b  # 내부적으로 add 함수가 호출됨

        # # 역전파 실행
        # c.backward()

        # print(a.grad)  # 출력: 1.0
        # print(b.grad)  # 출력: 1.0

        # print("running backward on", self)
        if self._ctx is None:
            return
        
        if self.grad is None and allow_fill:
            # fill in the first grad with one
            # this is "implicit gradient creation"
            
            assert self.data.size == 1
            # 순전파 결과:
            # [ 0.73245234 -1.52361438]

            # 그래디언트 초기화:
            # [[1. 1.]]
            self.grad = np.ones_like(self.data)
        
        assert(self.grad is not None)

        grads = self._ctx.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g is None:
                continue
            if g.shape != t.data.shape:
                print(f"grad shape must match tensor shape in {self._ctx}, {g.shape} != {t.data.shape}")
                assert(False)
            t.grad = g
            t.backward(False)
    
    def mean(self) -> "Tensor":
        div = Tensor(np.array([1/self.data.size], dtype=self.data.dtype))
        return self.sum().mul(div)
    
# An instaniation of the Function is the Context

class Function:
    def __init__(self, *tensors: "Tensor") -> None:
        self.parents: Tuple[Tensor, ...] = tensors
        self.saved_tensors: List[Any] = []
    
    def save_for_backward(self, *x: Any) -> None:
        # # extend() 사용 예제
        # list1 = [1, 2, 3]
        # list2 = [4, 5, 6]

        # # extend() 사용
        # list1.extend(list2)
        # print(list1)  # 출력: [1, 2, 3, 4, 5, 6]
        self.saved_tensors.extend(x)

    # note that due to how partialmethod works, self and arg are switched
    def apply(self, arg: Union["Tensor", Type["Function"]], *x: "Tensor") -> "Tensor":
        # ctx = arg(self, *x)
        # ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        # ret._ctx = ctx
        # return ret
        # print(type(arg) == Tensor) # to test
        if isinstance(arg, Tensor):
            op = self
            x = [arg]+list(x)
        else: 
            op = arg
            x = [self] + list(x)
        ctx = op(*x)
        ret = Tensor(op.forward(ctx, *[t.data for t in x]))
        ret._ctx = ctx
        return ret

def register(name: str, fxn: Type[Function]) -> None:
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

class Dot(Function):
    @staticmethod
    def forward(ctx: Function, input: np.ndarray, weight: np.ndarray) -> np.ndarray:
        # input is with a lot of 0's
        # weight is with random.uniform()
        ctx.save_for_backward(input, weight)
        return input.dot(weight)
    
    @staticmethod
    def backward(ctx: Function, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input, grad_weight

# def dot(self, other: "Tensor") -> "Tensor":
#     # 내부적으로 Dot.apply를 호출하며, 이때 self.data와 other.data를 전달합니다.
#     return self.dot_fn.apply(self, other)
register('dot', Dot)

class ReLU(Function):
    @staticmethod
    def forward(ctx: Function, input: np.ndarray) -> np.ndarray:
        # 순전파: 입력값을 저장하고 ReLU 연산 수행
        # it is not layer, just act for input 
        ctx.save_for_backward(input)
        return np.maximum(input, 0)
    
    @staticmethod
    def backward(ctx: Function, grad_output: np.ndarray) -> np.ndarray:
        # 역전파: 저장된 입력값을 사용하여 기울기 계산
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
register('relu',ReLU)

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Function, input: np.ndarray) -> np.ndarray:
        def logsumexp(x: np.ndarray) -> np.ndarray:
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
    def backward(ctx: Function, grad_output: np.ndarray) -> np.ndarray:
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape(-1, 1)

register('logsoftmax', LogSoftmax)

class Mul(Function):
    @staticmethod
    def forward(ctx: Function, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, y)
        return x*y
    
    @staticmethod
    def backward(ctx: Function, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def forward(ctx: Function, input: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(input)
        return np.array([input.sum()])
    
    @staticmethod
    def backward(ctx: Function, grad_output: np.ndarray) -> np.ndarray:
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)
register('sum', Sum)


class Add(Function):
    @staticmethod
    def forward(ctx: Function, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x+y

    @staticmethod 
    def backward(ctx: Function, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output
register('add', Add)

# from numba import jit, float32

class Conv2D(Function):
    @staticmethod
    def forward(ctx:Function, x:np.ndarray, w: np.ndarray) -> np.ndarray:
        # print(x.shape, 'x.shape')  # 입력 텐서 형태 디버깅
        # print(w.shape, 'w.shape')  # 커널 형태 디버깅
        # exit(0)  # 여기서 프로그램 종료 (현재 구현이 미완성임을 알림)
        
        ctx.save_for_backward(x, w)  # 역전파를 위해 입력 저장
        cout, cin, H, W = w.shape  # 커널 형태 해체 [출력채널, 입력채널, 높이, 너비]
        
        # 출력 텐서 초기화 (NCHW 형식)
        ret = np.zeros((x.shape[0], cout, x.shape[2]-(H-1), x.shape[3]-(W-1)), dtype=w.dtype)
        
        # 커널 재구성 [출력채널, 입력채널*H*W] 형태로 평탄화 후 전치
        tw = w.reshape(w.shape[0], -1).T
        
        # 출력 공간 순회 (높이와 너비 방향)
        for Y in range(ret.shape[2]):  # 출력 높이
            for X in range(ret.shape[3]):  # 출력 너비
                # 입력에서 커널 크기 영역 추출 (NCHW 형식 유지)
                tx = x[:, :, Y:Y+H, X:X+W].reshape(x.shape[0], -1)
                ret[:, :, Y, X] = tx.dot(tw)
        # for j in range(H):
        #     for i in range(W):
        #         tw = w[:, :, j, i] # 차원 축소됨, (1, 2)
        #         for Y in range(ret.shape[2]):
        #             for X in range(ret.shape[3]):
        #                 ret[:, :, Y, X] += x[:, :, Y+j, X+i].dot(tw.T)
        return ret

    @staticmethod
    def backward(ctx: Function, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # dx = np.zeros_like(x) # format 
        # dw = np.zeros_like(w) # format
        x, w = ctx.saved_tensors
        cout, cin, H, W = w.shape  #filter
        # for j in range(H):
        #     for i in range(W):
        #         tw = w[:, :, j, i]
        #         for Y in range(grad_output.shape[2]):
        #             for X in range(grad_output.shape[3]):
        #                 gg = grad_output[:, :, Y, X]
        #                 tx = x[:, :, Y, X]
        #                 dx[:, :, Y+j, X+j] += gg.dot(tw)
        #                 dw[:, :, j, i] += gg.T.dot(tx)
        dx, dw = np.zeros_like(x), np.zeros_like(w)
        tw = w.reshape(w.shape[0], -1)
        for Y in range(grad_output.shape[2]):
            for X in range(grad_output.shape[3]):
                gg = grad_output[:, :, Y, X]
                tx = x[:, :, Y:Y+H, X:X+W].reshape(x.shape[0], -1)
                dx[:, :, Y:Y+H, X:X+W] += gg.dot(tw).reshape(dx.shape[0], dx.shape[1], H, W)
                dw += gg.T.dot(tx).reshape(dw.shape)
        return dx, dw

    # @staticmethod
    # def forward(ctx: Function, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    #     ctx.save_for_backward(x, w)
    #     return Conv2D.inner_forward(x, w)
    # @staticmethod
    # def backward(ctx: Function, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     return Conv2D.inner_backward(grad_output, *ctx.saved_tensors)

register('conv2d', Conv2D)

class Reshape(Function):
    @staticmethod
    def forward(ctx: Function, x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)
    
    @staticmethod
    def backward(ctx: Function, grad_output: np.ndarray) -> Tuple[np.ndarray, Optional[Any]]:
        in_shape,  = ctx.saved_tensors
        return grad_output.reshape(in_shape), None 
register('reshape', Reshape)