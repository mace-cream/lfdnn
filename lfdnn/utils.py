import os
import struct
import numpy as np


class TensorOpUndefinedError(Exception):
    pass
class TensorOpNotSupported(Exception):
    pass


def one_hot(x, depth):
    if depth == 1:
        return x.reshape([len(x), -1])
    x_inner = np.array(x)
    result = np.matmul(np.ones((x_inner.shape[0], 1)), np.arange(depth).reshape((1, depth)))
    x_inner = np.matmul(x_inner.reshape((x_inner.shape[0], 1)), np.ones((1, depth)))
    return (result == x_inner) * 1.0

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _softmax(x):
    result = (np.exp(x).T / np.sum(np.exp(x), 1)).T
    if np.any(~np.isfinite(result)):
        result = (x-np.min(x))/(np.max(x)-np.min(x))
    return result
