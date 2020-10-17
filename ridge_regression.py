import numpy as np
from sklearn.metrics import r2_score

import lfdnn
from lfdnn import Graph, operator

class RidgeRegression(Graph):
    """
    ridge regression using Automatic differentiation
    Parameters
    ----------
    alpha: regularization parameter
    """
    def __init__(self, alpha=1.0, learning_rate=0.05, epoch_num=100, batch_size='auto'):
        # modify self.skip = False to run the extra test for bonus question
        self.skip = True
        self.alpha = alpha
        super().__init__(learning_rate=learning_rate, epoch_num=epoch_num, batch_size=batch_size)
        pass

    def construct_model(self, x_train, y_train):
        # put your code here

        # dummy acc
        self.accuracy = self.loss

    def fit(self, x_train, y_train):
        # alias for train
        self.train(x_train, y_train)

    def train(self, x_train, y_train):
        super().train(x_train, y_train)
        self.theta = self.weight_value['output_weight']
        self.b = self.weight_value['output_bias']

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
