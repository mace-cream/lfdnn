import numpy as np

from lfdnn.tensor import tensor, NameManager
from lfdnn.utils import _sigmoid, _softmax

NM = NameManager()

class matmul(tensor):
    '''
        matrix multiplication
    '''
    def __init__(self, x1, x2):
        super().__init__(x1.shape[:-1] + x2.shape[1:],
                    NM.get('matmul'), 'matmul', [x1, x2])
        x1.output_list.append(self)
        if x1 is not x2:
            x2.output_list.append(self)
    def eval(self, feed):
        result = np.matmul(self.input_list[0].eval(
                feed), self.input_list[1].eval(feed))
        feed.update({self.name: result})
        return result

class add(tensor):
    def __init__(self, x1, x2):
        super().__init__(x1.shape, NM.get('add'), 'add', [x1, x2])
        x1.output_list.append(self)
        if x1 is not x2:
            x2.output_list.append(self)
    def eval(self, feed):
        result = self.input_list[0].eval(feed) +\
                 self.input_list[1].eval(feed)
        feed.update({self.name: result})
        return result

class sigmoid(tensor):
    def __init__(self, x):
        super().__init__(x.shape,NM.get('sigmoid'), 'sigmoid', [x])
        x.output_list.append(self)
    def eval(self, feed):
        result = _sigmoid(self.input_list[0].eval(feed))
        feed.update({self.name: result})
        return result

class relu(tensor):
    def __init__(self, x):
        super().__init__(x.shape, NM.get('relu'), 'relu', [x])
        x.output_list.append(self)
    def eval(self, feed):
        result = self.input_list[0].eval(feed)
        result[result < 0] = 0
        feed.update({self.name: result})
        return result

class log(tensor):
    def __init__(self, x):
        super().__init__(x.shape, NM.get('log'), 'log', [x])
        x.output_list.append(self)
    def eval(self, feed):
        result = np.log(self.input_list[0].eval(feed))
        feed.update({self.name: result})
        return result
    
class product(tensor):
    '''
    elementwise multiplication of two tensors
    '''
    def __init__(self, x1, x2):
        super().__init__(x1.shape, NM.get('product'), 'product', [x1, x2])
        x1.output_list.append(self)
        if x1 is not x2:
            x2.output_list.append(self)
    def eval(self, feed):
        result = result = self.input_list[0].eval(
                feed)*self.input_list[1].eval(feed)
        feed.update({self.name: result})
        return result

def square_sum(x):
    out = reduce_mean(product(x, x))
    return out

class softmax(tensor):
    def __init__(self, x):
        super().__init__(x.shape, NM.get('softmax'), 'softmax', [x])
        x.output_list.append(self)
    def eval(self, feed):
        result = _softmax(self.input_list[0].eval(feed))
        feed.update({self.name: result})
        return result

class log_softmax(tensor):
    # we found that np.log(softmax(x)) have serious numerical issue.
    # therefore we define log_softmax() specifically.
    def __init__(self, x):
        super().__init__(x.shape, NM.get('log_softmax'), 'log_softmax', [x])
        x.output_list.append(self)
    def eval(self, feed):
        logit = self.input_list[0].eval(feed)
        # Note: The exact calculation of log(sum(exp(s_i))) has serious numerical issue, we use max instead.
        result = logit - np.log(np.sum(np.exp(logit), 1, keepdims=True))
        if np.any(~np.isfinite(result)):
            result = logit - np.max(logit, 1, keepdims=True)
        feed.update({self.name: result})
        return result
    
class reduce_sum(tensor):
    def __init__(self, x):
        super().__init__([1, 1], NM.get('reduce_sum'), 'reduce_sum', [x])
        x.output_list.append(self)
    def eval(self, feed):
        result = np.sum(self.input_list[0].eval(feed))
        feed.update({self.name: result})
        return result

class scale(tensor):
    '''multiply a tensor x by a scalar alpha

    Parameters
    ----------
    x: tensor object
    alpha: double
    '''
    def __init__(self, x, alpha):
        super().__init__(x.shape, NM.get('scale'), 'scale', [x, alpha])
        x.output_list.append(self)
    def eval(self, feed):
        result = self.input_list[1] * self.input_list[0].eval(feed)
        feed.update({self.name: result})
        return result

def reduce_mean(x):
    '''mean value of x along axis = 0
    '''
    out = scale(reduce_sum(x), 1 / x.shape[0])
    return out

def mse(x, y):
    '''mean square error
    '''
    subtract_out = add(x, scale(y, -1))
    out = square_sum(subtract_out)
    return out

def CE(x, y):
    '''average cross-entropy multiplied by -1
    see the explanation: https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
    '''
    out = scale(reduce_mean(product(y, log(x))), -1)
    return out

def CE_with_logit(x, y):
    '''loss function for multi-class classification
    '''
    out = scale(reduce_mean(product(y, log_softmax(x))), -1)
    return out

class accuracy(tensor):
    '''the fraction of right prediction
    '''
    def __init__(self, pred, y):
        super().__init__([1,1], NM.get('accuracy'), 'accuracy', [pred, y])
        pred.output_list.append(self)
        y.output_list.append(self)
    def eval(self, feed):
        result = np.mean(np.argmax(self.input_list[0].eval(
                feed), -1) == np.argmax(self.input_list[1].eval(feed), -1))
        feed.update({self.name: result})
        return result
