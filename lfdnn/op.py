from lfdnn.tensor import tensor, NameManager
import numpy as np

NM = NameManager()

def matmul(x1, x2):
    out = tensor(x1.shape[:-1] + x2.shape[1:],
                 NM.get('matmul'), 'matmul', [x1, x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def add(x1, x2):
    out = tensor(x1.shape, NM.get('add'), 'add', [x1, x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def sigmoid(x):
    out = tensor(x.shape,NM.get('sigmoid'), 'sigmoid', [x])
    x.output_list.append(out)
    return out

def relu(x):
    out = tensor(x.shape, NM.get('relu'), 'relu', [x])
    x.output_list.append(out)
    return out

def log(x):
    out = tensor(x.shape, NM.get('log'), 'log', [x])
    x.output_list.append(out)
    return out

def product(x1, x2):
    '''
    elementwise multiplication of two tensors
    '''
    out = tensor(x1.shape, NM.get('product'), 'product', [x1, x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def softmax(x):
    out = tensor(x.shape, NM.get('softmax'), 'softmax', [x])
    x.output_list.append(out)
    return out

def log_softmax(x):
    # we found that np.log(softmax(x)) have serious numerical issue.
    # therefore we define log_softmax() specifically.
    out = tensor(x.shape, NM.get('log_softmax'), 'log_softmax', [x])
    x.output_list.append(out)
    return out

def reduce_sum(x):
    out = tensor([1,1], NM.get('reduce_sum'), 'reduce_sum', [x])
    x.output_list.append(out)
    return out

def scale(x, alpha):
    '''
        multiply a tensor x by a scalar alpha
    '''
    out = tensor(x.shape, NM.get('scale'), 'scale', [x, alpha])
    x.output_list.append(out)
    return out

def reduce_mean(x):
    out = scale(reduce_sum(x), 1 / x.shape[0])
    return out

def CE(x, y):
    '''
    average cross-entropy multiplied by -1
    '''
    out = scale(reduce_mean(product(y, log(x))), -1)
    return out

def CE_with_logit(x, y):
    out = scale(reduce_mean(product(y, log_softmax(x))), -1)
    return out

def accuracy(pred, y):
    '''
        the fraction of right prediction
    '''
    out = tensor([1,1], NM.get('accuracy'), 'accuracy', [pred, y])
    pred.output_list.append(out)
    y.output_list.append(out)
    return out



