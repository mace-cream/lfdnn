import numpy as np

from lfdnn.tensor import tensor, NameManager
from lfdnn.utils import _sigmoid, _softmax

# initialize the global tensor name manager
NM = NameManager()

class matmul(tensor):
    '''tensor for matrix multiplication

    Parameters
    ----------
    x1: tensor, input tensor 1
    x2: tensor, input tensor 2
    '''
    def __init__(self, x1, x2):
        super().__init__(x1.shape[:-1] + x2.shape[1:],
                    NM.get('matmul'), [x1, x2])
        x1.output_list.append(self)
        if x1 is not x2:
            x2.output_list.append(self)
    def _eval(self, feed):
        result = np.matmul(self.input_list[0].eval(
                feed), self.input_list[1].eval(feed))
        return result
    def _derivative(self, feed, input, target):
        gradient = 0
        if input is self.input_list[0]:
            gradient = np.matmul(self.back(target, feed),
                            self.input_list[1].eval(feed).T)
        if input is self.input_list[1]:
            gradient += np.matmul(self.input_list[0].eval(
                    feed).T, self.back(target, feed))
        return gradient

class add(tensor):
    '''tensor for matrix add

    Parameters
    ----------
    x1: tensor, input tensor 1
    x2: tensor, input tensor 2
    '''    
    def __init__(self, x1, x2):
        super().__init__(x1.shape, NM.get('add'), [x1, x2])
        x1.output_list.append(self)
        if x1 is not x2:
            x2.output_list.append(self)
    def _eval(self, feed):
        result = self.input_list[0].eval(feed) + \
                 self.input_list[1].eval(feed)
        return result
    def _derivative(self, feed, input, target):
        dim_difference = len(self.shape) - len(input.shape)
        boardcast_dim = [i for i in range(dim_difference)] + [i for i in range(
            dim_difference, len(self.shape)) if input.shape[i - dim_difference] == 1]
        gradient = 0
        if input is self.input_list[0]:
            gradient = np.sum(self.back(target, feed), tuple(
                    boardcast_dim)).reshape(input.shape)
        if input is self.input_list[1]:
            gradient += np.sum(self.back(target, feed), tuple(
                    boardcast_dim)).reshape(input.shape)
        return gradient

class sigmoid(tensor):
    '''tensor for matrix elementwise sigmoid activation

    Parameters
    ----------
    x: tensor, input tensor
    '''    
    def __init__(self, x):
        super().__init__(x.shape,NM.get('sigmoid'), [x])
        x.output_list.append(self)
    def _eval(self, feed):
        result = _sigmoid(self.input_list[0].eval(feed))
        return result
    def _derivative(self, feed, input, target):
        # write down the derivative of sigmoid here
        jacob = 1
        # end of your writing
        return jacob * self.back(target, feed)
        
class relu(tensor):
    '''tensor for matrix elementwise relu activation

    Parameters
    ----------
    x: tensor, input tensor
    '''    
    def __init__(self, x):
        super().__init__(x.shape, NM.get('relu'), [x])
        x.output_list.append(self)
    def _eval(self, feed):
        result = self.input_list[0].eval(feed)
        result[result < 0] = 0
        return result
    def _derivative(self, feed, input, target):
        # input must be in self.input_list
        local_gradient = self.eval(feed)
        local_gradient = (local_gradient > 0) * 1.0
        return local_gradient * self.back(target, feed)

class log(tensor):
    '''tensor for matrix elementwise log function

    Parameters
    ----------
    x: tensor, input tensor
    '''        
    def __init__(self, x):
        super().__init__(x.shape, NM.get('log'), [x])
        x.output_list.append(self)
    def _eval(self, feed):
        result = np.log(self.input_list[0].eval(feed))
        return result
    def _derivative(self, feed, input, target):
        return 1 / input.eval(feed) * self.back(target, feed)

class product(tensor):
    '''tensor for matrix elementwise multiplication

    Parameters
    ----------
    x1: tensor, input tensor 1
    x2: tensor, input tensor 2
    '''
    def __init__(self, x1, x2):
        super().__init__(x1.shape, NM.get('product'), [x1, x2])
        x1.output_list.append(self)
        if x1 is not x2:
            x2.output_list.append(self)
    def _eval(self, feed):
        # write down the evaluation of product here
        # you should modify the following line
        result = self.input_list[0].eval(feed)
        # end of your writing
        return result
    def _derivative(self, feed, input, target):
        gradient = 0
        if input is self.input_list[0]:
            gradient = self.back(target, feed) * self.input_list[1].eval(feed)
        if input is self.input_list[1]:
            gradient += self.back(target, feed) * self.input_list[0].eval(feed)
        return gradient

class softmax(tensor):
    '''tensor for matrix softmax function

    Parameters
    ----------
    x: tensor, input tensor
    '''
    def __init__(self, x):
        super().__init__(x.shape, NM.get('softmax'), [x])
        x.output_list.append(self)
    def _eval(self, feed):
        result = _softmax(self.input_list[0].eval(feed))
        return result

class log_softmax(tensor):
    '''tensor for log softmax function

    Parameters
    ----------
    x: tensor, input tensor
    '''
    def __init__(self, x):
        super().__init__(x.shape, NM.get('log_softmax'), [x])
        x.output_list.append(self)
    def _eval(self, feed):
        logit = self.input_list[0].eval(feed)
        result = logit - np.log(np.sum(np.exp(logit), 1, keepdims=True))
        return result
    def _derivative(self, feed, input, target):
        logits = _softmax(input.eval(feed))
        forward_gradient = self.back(target, feed)
        local_gradient = []
        for i in range(input.shape[0]):
            local_logits = logits[i].reshape((-1, 1))
            jacob = np.eye(
                local_logits.shape[0]) - np.matmul(local_logits, (0 * local_logits + 1).T)
            local_gradient.append(
                np.matmul(jacob, forward_gradient[i].reshape((-1, 1))).T)
        local_gradient = np.concatenate(local_gradient, 0)
        return local_gradient

class reduce_sum(tensor):
    '''tensor for reduced sum

    Notice: this is a special tensor since its `_eval` function gives a numerical
    value instead of a matrix.

    Parameters
    ----------
    x: tensor, input tensor
    '''
    def __init__(self, x):
        super().__init__([1, 1], NM.get('reduce_sum'), [x])
        x.output_list.append(self)
    def _eval(self, feed):
        # reduced sum along all axises
        result = np.sum(self.input_list[0].eval(feed))
        return result
    def _derivative(self, feed, input, target):
        return np.ones(input.shape) * self.back(target, feed)

class scale(tensor):
    '''tensor for scalar multiplication, alpha * x

    Parameters
    ----------
    x: tensor object
    alpha: double
    '''
    def __init__(self, x, alpha):
        super().__init__(x.shape, NM.get('scale'), [x, alpha])
        x.output_list.append(self)
    def _eval(self, feed):
        result = self.input_list[1] * self.input_list[0].eval(feed)
        return result
    def _derivative(self, feed, input, target):
        return self.input_list[1] * self.back(target, feed)

class accuracy(tensor):
    '''tensor for accuracy, used in multi-class classification task

    Parameters
    ----------
    pred: tensor, prediction results
    y: tensor, true label
    '''
    def __init__(self, pred, y):
        super().__init__([1,1], NM.get('accuracy'), [pred, y])
        pred.output_list.append(self)
        y.output_list.append(self)
    def _eval(self, feed):
        result = np.mean(np.argmax(self.input_list[0].eval(
                feed), -1) == np.argmax(self.input_list[1].eval(feed), -1))
        return result

def mean_square_sum(x):
    '''mean square sum, or square of Frobenius norm

    Parameters
    ----------
    x: tensor

    Returns
    -------
    tensor object    
    '''
    out = reduce_mean(product(x, x))
    return out

def reduce_mean(x):
    '''mean value of x along axis = 0

    Parameters
    ----------
    x: tensor

    Returns
    -------
    tensor object
    '''
    out = scale(reduce_sum(x), 1.0 / x.shape[0])
    return out

def mse(x, y):
    '''mean square error

    Parameters
    ----------
    x: tensor
    y: tensor

    Returns
    -------
    tensor object, which is the mean squared error of x
    '''
    # put your composition model here
    out = reduce_mean(x)
    # end of your writing
    return out

def CE(x, y):
    '''average cross-entropy multiplied by -1
    for detail, see the explanation:
    https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression

    Parameters
    ----------
    x: tensor
    y: tensor

    Returns
    -------
    tensor object
    '''
    out = scale(reduce_mean(product(y, log(x))), -1)
    return out

def CE_with_logit(x, y):
    '''loss function for multi-class classification
       logit means the output part before the softmax

    Parameters
    ----------
    x: tensor
    y: tensor

    Returns
    -------
    tensor object
    '''
    out = scale(reduce_mean(product(y, log_softmax(x))), -1)
    return out



