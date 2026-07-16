"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features,
                                           device=device, dtype=dtype))
        if bias:
            bias = init.kaiming_uniform(fan_in=out_features, fan_out=1,
                                         device=device, dtype=dtype)
            self.bias = Parameter(ops.reshape(bias, (1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mul = ops.matmul(X, self.weight)
        if self.bias is None:
            return mul
        shape = [1]*(len(mul.shape)-1)+[self.out_features]
        return ops.add(mul, ops.broadcast_to(self.bias.reshape(shape), mul.shape))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        old_shape = X.shape
        combine_dimension = 1
        for i in old_shape[1:]:
            combine_dimension = combine_dimension*i
        new_shape = [old_shape[0], combine_dimension]
        return ops.reshape(X, new_shape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_classes = logits.shape[1]
        y_one_hot = init.one_hot(num_classes, y, device=logits.device)
        batch_size = logits.shape[0]
        lse = ops.logsumexp(logits, axes=(1,))
        z_y = ops.summation(logits * y_one_hot, axes=(1,))
        return ops.summation(lse - z_y) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, requires_grad=False, device=device, dtype=dtype)
        self.running_var = init.ones(dim, requires_grad=False, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        if self.training:
            expect_x = ops.divide_scalar(ops.summation(x, axes=(0, )), batch_size)
            self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*expect_x
        else:
            expect_x = self.running_mean
        expect_x = ops.broadcast_to(ops.reshape(expect_x, (1, self.dim)), x.shape)
        up = x-expect_x

        if self.training:
            variance_x = ops.summation(ops.power_scalar(up, 2), axes=(0, ))
            variance_x = ops.divide_scalar(variance_x, batch_size)
            self.running_var = (1-self.momentum)*self.running_var+self.momentum*variance_x
        else:
            variance_x = self.running_var
        below = ops.power_scalar(ops.add_scalar(variance_x, self.eps), 0.5)
        below = ops.broadcast_to(ops.reshape(below, (1, self.dim)), x.shape)

        normal_x = ops.divide(up, below)
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return normal_x*broadcast_weight+broadcast_bias
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ndim = len(x.shape)
        expect_x = ops.divide_scalar(ops.summation(x, axes=(ndim-1, )), self.dim)
        mean_shape = list(x.shape)
        mean_shape[-1] = 1
        expect_x = ops.broadcast_to(ops.reshape(expect_x, tuple(mean_shape)), x.shape)
        up = x-expect_x

        variance_x = ops.reshape(ops.summation(ops.power_scalar(up, 2), axes=(ndim-1, )), mean_shape)
        variance_x = ops.divide_scalar(variance_x, self.dim)
        below = ops.power_scalar(ops.add_scalar(variance_x, self.eps), 0.5)
        below = ops.broadcast_to(below, x.shape)

        normal_x = ops.divide(up, below)
        weight_shape = [1] * (ndim - 1) + [self.dim]
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, weight_shape), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, weight_shape), x.shape)
        return normal_x*broadcast_weight+broadcast_bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=(1-self.p), device=x.device, dtype=x.dtype)
            x = ops.divide_scalar(ops.multiply(x, mask), 1-self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.add(x, self.fn.forward(x))
        ### END YOUR SOLUTION
