import numpy as np
from sklearn.metrics import accuracy_score

import lfdnn
from lfdnn import Graph, operator

def _svm_loss(X, y, w, b, C):
    term_1 = np.linalg.norm(w) ** 2 / 2
    term_2 = 1 - y * (X @ w + b)
    term_2 = (term_2 + np.abs(term_2)) / 2
    return term_1 + np.sum(term_2) * C

class SVM(Graph):
    """
    soft margin svm

    Parameters
    ----------
    C: double, positive number for regularization
    learning_rate: double, learning rate in SGD
    epoch_num: int, number of iteration
    batch_size: int, batch size used in SGD, default to all data
    """
    def __init__(self, C=1.0, learning_rate=0.05, epoch_num=100, batch_size='auto'):
        # modify self.skip = False to run the extra test for bonus question
        self.skip = True
        self.C = C
        super().__init__(learning_rate=learning_rate, epoch_num=epoch_num, batch_size=batch_size)
        pass

    def construct_model(self, x_train, y_train):
        # get number of features
        input_dim = x_train.shape[-1]
        # get number of classes
        output_dim = 1
        batch_size = self.batch_size
        if batch_size == 'auto':
            # use all data
            batch_size = x_train.shape[0]

        self.input = lfdnn.tensor([batch_size, input_dim], 'input')
        self.label = lfdnn.tensor([batch_size, output_dim], 'label')
        w = lfdnn.tensor([input_dim, output_dim], 'output_weight')
        self.weight['output_weight'] = w
        b = lfdnn.tensor([1, output_dim], 'output_bias')
        self.weight['output_bias'] = b        
        # put your code here, you can adjust the following lines
        self.output = operator.add(operator.matmul(self.input, w), b)
        self.loss = self.output
        # end of your modification
        # dummy acc
        self.accuracy = self.loss
        
    def fit(self, x_train, y_train):
        # alias for train
        self.train(x_train, y_train)

    def predict(self, x_test):
        """Predict class labels for samples in x_test

        Parameters
        ----------
        x_test: np.array, shape (num_samples, num_features)

        Returns
        -------
        pred: np.array, shape (num_samples, )
        """
        num_of_data = x_test.shape[0]
        classes_ = (x_test @ self.w + self.b ) > 0
        classes = 2 * classes_ - 1
        return classes

    def train(self, x_train, y_train):
        super().train(x_train, y_train, verbose=False)
        candidate_w = self.weight_value['output_weight']
        self.w = candidate_w.reshape([candidate_w.shape[0]])
        candidate_b = self.weight_value['output_bias']
        self.b = candidate_b.reshape([candidate_b.shape[0]])

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)