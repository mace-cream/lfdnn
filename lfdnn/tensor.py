import numpy as np
import os

from lfdnn.utils import _sigmoid, _softmax
from lfdnn.utils import one_hot

class tensor(object):
    def __init__(self, shape, name, input_list=None, value=None):
        self.shape = shape
        self.name = name
        self.input_list = input_list
        self.output_list = []
        self.value = value

    def forward(self, feed):
        '''
        evaluate the function given feed data
        '''
        return self.eval(feed)

    def differentiate(self, variable, feed):
        '''
        calculate the derivate about `variable` at the given feed data
        '''
        return variable.back(self, feed)

    def eval(self, feed):
        '''
        Define the forward computation given input 'feed'
        '''
        if self.name in feed.keys():
            return feed[self.name]
        result = self._eval(feed)
        feed.update({self.name: result})
        return result

    def _eval(self, feed):
        return 0

    def _derivative(self, feed, input, target):
        return 0

    def back(self, target, feed):
        '''Define the gradient back propagation with respect to 'target' given input 'feed'
        '''
        if self.name + '_g' in feed.keys():
            return feed[self.name+'_g']
        if self is target:
            return np.ones(self.shape)
        gradient = 0
        for out in self.output_list:
            gradient += out._derivative(feed, self, target)

        feed.update({self.name+'_g': gradient})
        return gradient


class NameManager(object):
    def __init__(self):
        self.nameList = {}

    def get(self, name):
        if name not in self.nameList.keys():
            self.nameList.update({name: 0})
        else:
            self.nameList.update({name: self.nameList[name] + 1})
        return name + '_' + str(self.nameList[name])


class Graph:
    '''base class for machine learning objective function
       the optimizer is based on stochastic gradient descent
    '''
    def __init__(self, learning_rate=0.05, epoch_num=1, batch_size='auto'):
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.weight = {}
        self.weight_value = {}
        # the following members should be defined to proper tensor in `construct_model`
        self.input = None
        self.output = None # y_predict
        self.label = None # y_train input
        self.loss = None
        self.accuracy = None

    def construct_model(self, x_train, y_train):
        '''this function should be overridden to provide actual construction code
        '''
        raise NotImplementedError("base class function `construct_model` not callable")

    def initWeight(self, initializer=np.random.standard_normal):
        self.weight_value = {k: initializer(
            v.shape) for k, v in self.weight.items()}

    def update(self, feed):
        gradient = {k: v.back(self.loss, feed) for k, v in self.weight.items()}
        self.weight_value.update({
            k: self.weight_value[k]- self.learning_rate * gradient[k] for k in self.weight.keys()})

    def predict(self, x_test):
        feed = {self.input.name:  x_test}
        feed.update(self.weight_value)
        return self.output.eval(feed)

    def _epoch_iterate(self, x_batch, y_batch):
        '''return the loss and accuracy of current epoch
        '''
        feed = {self.input.name: x_batch, self.label.name: y_batch}
        feed.update(self.weight_value)
        loss_val = self.loss.eval(feed)
        acc = self.accuracy.eval(feed)
        self.update(feed)
        return (loss_val, acc)
        
    def train(self, x_train, y_train, verbose=False):
        self.construct_model(x_train, y_train)
        self.initWeight()
        output_dim = self.label.shape[-1]
        batch_size = self.input.shape[0]
        for _ in range(self.epoch_num):
            counter = 0
            while counter + batch_size <= x_train.shape[0]:
                x_batch = x_train[counter: counter + batch_size].reshape([batch_size, -1])
                y_batch = one_hot(y_train[counter: counter + batch_size], output_dim)
                loss_val, acc = self._epoch_iterate(x_batch, y_batch)
                if verbose:
                    print(loss_val, acc)
                counter += batch_size