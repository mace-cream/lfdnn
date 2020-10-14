import os
import struct
import numpy as np

class TensorOpUndefinedError(Exception):
    pass
class TensorOpNotSupported(Exception):
    pass

def one_hot(x,depth):
    result = np.matmul(np.ones((x.shape[0],1)),np.arange(depth).reshape((1,depth)))
    x = np.matmul(x.reshape((x.shape[0],1)),np.ones((1,depth)))
    return (result==x)*1.0

def _sigmoid(x):
    return 1/(1+np.exp(-x))

def _softmax(x):
    result = (np.exp(x).T/np.sum(np.exp(x),1)).T
    if np.any(~np.isfinite(result)):
        result = (x-np.min(x))/(np.max(x)-np.min(x))
    return result
