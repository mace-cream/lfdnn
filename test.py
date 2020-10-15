import unittest
import copy

import numpy as np
from sklearn.datasets import load_iris

from lfdnn import tensor
from lfdnn import sigmoid, add, reduce_mean, product, matmul, CE_with_logit, scale

from lfdnn.utils import TensorOpUndefinedError, TensorOpNotSupported
from lfdnn.numerical_tensor import numerical_accuracy

from model import MLP

class TestAutoDifferential(unittest.TestCase):
    def test_shape(self):
        a = tensor([1, 2], 'a')
        self.assertEqual(reduce_mean(a).shape, [1, 1])

    def test_forward(self):
        a = tensor([1, 2], 'a')
        feed = {'a': np.array([[5, 6]])}
        self.assertAlmostEqual(reduce_mean(a).forward(feed), 5.5)

    def test_backward(self):
        a = tensor([1, 1], 'a')
        feed = {'a': np.array([[0]])}
        self.assertAlmostEqual(sigmoid(a).differentiate(a, feed), 0.25)

    def test_null_operator(self):
        a = tensor([1, 2], 'a')
        feed = {}
        with self.assertRaises(TensorOpUndefinedError):
            a.forward(feed)

    def test_unsupported_operator(self):
        a = tensor([1, 2], 'a', op_type='subtract')
        feed = {'b': np.array([[5, 6]])}
        with self.assertRaises(TensorOpNotSupported):
            a.forward(feed)

    def test_numerical_accuracy(self):
        prob_vector = np.array([[0.3, 0.7], [0.6, 0.4]])
        true_vector = [1, 1]
        self.assertAlmostEqual(numerical_accuracy(prob_vector, true_vector), 0.5)
        true_vector = [1, 0]
        self.assertAlmostEqual(numerical_accuracy(prob_vector, true_vector), 1.0)

class TestMLP(unittest.TestCase):
    def test_construction_model(self):
        mlp = MLP()
        # number of data = 3
        # number of feature = 2
        x_train = np.zeros([3, 2])
        y_train = [1, 0, 1]
        mlp.train(x_train, y_train)
    def test_iris(self):
        # test softmax on Iris dataset
        iris = load_iris()
        x_train = iris.data
        batch_size = int(x_train.shape[0] / 3)
        y_train = iris.target
        mlp = MLP(epoch_num=400, batch_size=batch_size, learning_rate=0.1)
        np.random.seed(2020)
        mlp.train(x_train, y_train)
        y_predict = mlp.predict(x_train)
        self.assertTrue(numerical_accuracy(y_predict, y_train) > 0.95)
        # save the results for plotting
        
if __name__=="__main__":
    unittest.main()
