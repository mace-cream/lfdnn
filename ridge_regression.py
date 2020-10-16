import numpy as np
from sklearn.metrics import r2_score

import lfdnn
from lfdnn import Graph, operator

class RidgeRegression(Graph):
    """
    ridge regression using Automatic differentiation
    Parameters
    ----------
    alpha: regularization strength
    """
    def __init__(self, alpha=1.0, learning_rate=0.05, epoch_num=1, batch_size='auto'):
        self.alpha = alpha
        super().__init__(learning_rate=learning_rate, epoch_num=epoch_num, batch_size=batch_size)
        pass

    def construct_model(self, x_train, y_train):
        # get number of features
        input_dim = x_train.shape[-1]
        # get number of classes
        output_dim = len(np.unique(y_train))
        batch_size = self.batch_size
        _lambda = self.alpha
        if batch_size == 'auto':
            # use all data
            batch_size = x_train.shape[0]

        self.input = lfdnn.tensor([batch_size, input_dim], 'Input')
        self.target_value = lfdnn.tensor([batch_size, output_dim], 'target_value')
        h = self.input
        w = lfdnn.tensor([input_dim, output_dim], 'output_weight')
        self.weight['output_weight'] = w
        b = lfdnn.tensor([1, output_dim], 'output_bias')
        self.weight['output_bias'] = b
        h = operator.add(operator.matmul(h, w), b)
        self.output = h
        self.loss = operator.mse(h, self.target_value)
        if _lambda > 0:
            for w in self.weight.values():
                regularization_term = operator.scale(operator.square_sum(w), _lambda)
                self.loss = operator.add(self.loss, regularization_term)
        # dummy acc
        self.accuracy = self.loss

    def fit(self, x_train, y_train):
        # alias for train
        self.train(x_train, y_train)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
