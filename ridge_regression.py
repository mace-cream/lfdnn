import numpy as np
from sklearn.metrics import r2_score

import lfdnn
from lfdnn import Graph

class RidgeRegression(Graph):
    """
    ridge regression using Automatic differentiation
    Parameters
    ----------
    alpha: regularization strength
    """
    def __init__(self, alpha=1.0 , learning_rate=0.05, epoch_num=1, batch_size='auto'):
        self.alpha = alpha
        super().__init__(learning_rate=learning_rate, epoch_num=epoch_num, batch_size=batch_size)
        pass

    def construct_model(self, x_train, y_train):
        # get number of features
        InputDim = x_train.shape[-1]
        # get number of classes
        OutputDim = len(np.unique(y_train))
        BatchSize = self.batch_size
        _lambda = self.alpha
        if BatchSize == 'auto':
            # use all data
            BatchSize = x_train.shape[0]

        self.input = lfdnn.tensor([BatchSize, InputDim], 'Input')
        self.label = lfdnn.tensor([BatchSize, OutputDim], 'Label')
        h = self.input
        w = lfdnn.tensor([InputDim, OutputDim], 'output_weight')
        self.weight['output_weight'] = w
        b = lfdnn.tensor([1, OutputDim], 'output_bias')
        self.weight['output_bias'] = b
        self.output = lfdnn.add(lfdnn.matmul(h, w), b)
        self.loss = lfdnn.CE_with_logit(h, self.label)
        if _lambda > 0:
            for w in self.weight.values():
                self.loss = lfdnn.add(self.loss, lfdnn.scale(lfdnn.reduce_mean(lfdnn.product(w, w)), _lambda))
        self.accuracy = lfdnn.accuracy(self.output, self.label)

    def fit(self, x_train, y_train):
        # alias for train
        self.train(x_train, y_train)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
