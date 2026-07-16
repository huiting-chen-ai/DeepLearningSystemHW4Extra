"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad*self.scalar*(a**(self.scalar-1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad/b, -out_grad*a/(b**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_axes = [i for i in range(len(a.shape))]
        if self.axes is not None:
            new_axes[self.axes[0]] = self.axes[1]
            new_axes[self.axes[1]] = self.axes[0]
        else:
            temp = new_axes[-1]
            new_axes[-1] = new_axes[-2]
            new_axes[-2] = temp
        return a.permute(new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        original_shape = node.inputs[0].shape
        return out_grad.reshape(original_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = self.shape
        sum_axes = []
        padded_input_shape = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
        for i, (in_dim, out_dim) in enumerate(zip(padded_input_shape, output_shape)):
            if in_dim != out_dim:
                sum_axes.append(i)
        grad = out_grad.sum(tuple(sum_axes))
        return grad.reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        new_shape = list(input_shape)
        if self.axes is not None:
            if isinstance(self.axes, int):
                new_shape[self.axes] = 1
            else:
                for axis in self.axes:
                    new_shape[axis] = 1
        else:
            new_shape = [1]*len(input_shape)
        return out_grad.reshape(new_shape).broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a@b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a, grad_b = out_grad@transpose(b), transpose(a)@out_grad
        if grad_a.shape != a.shape:
            axes = tuple(range(len(grad_a.shape) - len(a.shape)))
            grad_a = grad_a.sum(axes)
        if grad_b.shape != b.shape:
            axes = tuple(range(len(grad_b.shape) - len(b.shape)))
            grad_b = grad_b.sum(axes)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return numpy.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad/a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return numpy.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return exp(a)*out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        result = array_api.maximum(a, 0)
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad*Tensor(a>0, device=node.inputs[0].device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return numpy.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad*(1-(numpy.tanh(a))**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        target_shape = list(args[0].shape)
        target_shape = target_shape[:self.axis]+[len(args)]+target_shape[self.axis:]
        result = array_api.empty(tuple(target_shape), device=args[0].device)
        slices = [slice(0, s) for s in target_shape]
        for i, arg in enumerate(args):
            slices = [slice(None)] * len(target_shape)
            slices[self.axis] = i
            result[tuple(slices)] = arg
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        input_shape = list(A.shape)
        output_shape = input_shape[:self.axis]+input_shape[self.axis+1:]
        result = []
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(input_shape)
            slices[self.axis] = i
            result.append(A[tuple(slices)].compact().reshape(output_shape))
        return tuple(result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(out_grad.realize_cached_data(), self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = tuple(s*(self.dilation+1) if i in self.axes else s for i, s in enumerate(a.shape))
        result = array_api.full(new_shape, 0, device=a.device)
        slices = tuple(slice(None, None, self.dilation+1) if i in self.axes else slice(None, None, None) for i in range(len(a.shape)))
        result[slices] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = tuple(s//(self.dilation+1) if i in self.axes else s for i, s in enumerate(a.shape))
        result = array_api.empty(new_shape, device=a.device)
        slices = tuple(slice(None, None, self.dilation+1) if i in self.axes else slice(None, None, None) for i in range(len(a.shape)))
        result = a[slices]
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.compact()
        B = B.compact()
        if self.padding != 0:
            A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        new_kernel = B.reshape((K*K*C_in, C_out)).compact()
        new_input_shape = (N, (H-K+1)//self.stride, (W-K+1)//self.stride, K, K, C_in)
        new_input_stride = (A._strides[0], A._strides[1]*self.stride, A._strides[2]*self.stride, A._strides[1], A._strides[2], A._strides[3])
        new_input = A.as_strided(new_input_shape, new_input_stride).compact()
        new_input = new_input.reshape((N*((H-K+1)//self.stride)*((W-K+1)//self.stride), K*K*C_in)).compact()
        out = new_input@new_kernel
        out = out.reshape((N, (H-K+1)//self.stride, (W-K+1)//self.stride, C_out)).compact()
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs[0], node.inputs[1]
        B_flip = flip(B, (0, 1))
        B_flip = transpose(B_flip, (2, 3))
        out_grad_dilate = dilate(out_grad, (1, 2), self.stride-1)
        A_grad = conv(out_grad_dilate, B_flip, stride=1, padding=B.shape[0]-1-self.padding)

        A_permute = transpose(A, (0, 3))
        out_grad_dilate__ = transpose(out_grad_dilate, (0, 1))
        out_grad_dilate__ = transpose(out_grad_dilate__, (1, 2))
        B_grad = conv(A_permute, out_grad_dilate__, stride=1, padding=self.padding)
        B_grad = transpose(B_grad, (0, 1))
        B_grad = transpose(B_grad, (1, 2))
        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


