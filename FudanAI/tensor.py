import numpy as np
from typing import Union, Tuple

Arrayable = Union[float, list, np.ndarray]


def ensure_ndarray(data: Arrayable) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


def ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)


class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool = False, grad_fn=None):
        self.data = ensure_ndarray(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> "Tensor":
        from .autograd.functional import add

        return add(self, ensure_tensor(other))

    def __radd__(self, other) -> "Tensor":
        from .autograd.functional import add

        return add(ensure_tensor(other), self)

    def __iadd__(self, other) -> "Tensor":
        self.data += ensure_tensor(other).data
        return self

    def __neg__(self) -> "Tensor":
        from .autograd.functional import neg

        return neg(self)

    def __sub__(self, other) -> "Tensor":
        from .autograd.functional import sub

        return sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        from .autograd.functional import sub

        return sub(ensure_tensor(other), self)

    def __isub__(self, other) -> "Tensor":
        self.data -= ensure_tensor(other).data
        return self

    def __mul__(self, other) -> "Tensor":
        from .autograd.functional import mul

        return mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> "Tensor":
        from .autograd.functional import mul

        return mul(ensure_tensor(other), self)

    def __truediv__(self, other) -> "Tensor":
        from .autograd.functional import div

        return div(self, ensure_tensor(other))

    def __rtruediv__(self, other) -> "Tensor":
        from .autograd.functional import div

        return div(ensure_tensor(other), self)

    def __matmul__(self, other) -> "Tensor":
        from .autograd.functional import matmul

        return matmul(self, other)

    def __pow__(self, other: float) -> "Tensor":
        from .autograd.functional import pow

        return pow(self, other)

    def sum(self, axis: Union[int, Tuple[int]] = None) -> "Tensor":
        from .autograd.functional import sum

        return sum(self, axis)

    def mean(self, axis: Union[int, Tuple[int]] = None) -> "Tensor":
        from .autograd.functional import mean

        return mean(self, axis)

    def t(self) -> "Tensor":
        """transpose"""
        from .autograd.functional import t

        return t(self)

    def exp(self) -> "Tensor":
        from .autograd.functional import exp

        return exp(self)

    def relu(self) -> "Tensor":
        from .autograd.functional import relu

        return relu(self)

    def backward(self, grad: "Tensor" = None) -> None:
        from .autograd.engine import Engine

        assert self.requires_grad
        if grad is None and self.shape != ():
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        grad = grad if grad else Tensor(1.0)

        engine = Engine()
        engine.execute(self, grad)

    def zero_grad(self) -> None:
        self.grad = None


def rand(*shape, requires_grad=False) -> Tensor:
    data = np.random.randn(*shape)
    return Tensor(data=data, requires_grad=requires_grad)
