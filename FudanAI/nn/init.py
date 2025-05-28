from FudanAI import Tensor

import numpy as np
from .parameter import Parameter


def normal_(tensor: Parameter, mean=0.0, std=1.0):
    """
    Fills the input Tensor with values drawn from a normal distribution
    with given mean and standard deviation.
    """
    tensor.data = np.random.normal(mean, std, tensor.data.shape)
    return tensor


def xavier_uniform_(tensor: Parameter, gain=1.0):
    """
    Fills the input Tensor with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
    """
    fan_in, fan_out = tensor.data.shape
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    a = np.sqrt(3.0) * std
    tensor.data = np.random.uniform(-a, a, tensor.data.shape)
    return tensor
