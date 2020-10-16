import unittest
import copy

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.multiclass import OneVsRestClassifier
# scikit-learn >= 0.23 has this private API `_logistic_loss`
from sklearn.linear_model._logistic import _logistic_loss
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal

from lfdnn import tensor
from lfdnn import sigmoid, add, reduce_mean, product, matmul, CE_with_logit, scale

from lfdnn.utils import TensorOpUndefinedError, TensorOpNotSupported
from lfdnn.numerical_tensor import numerical_accuracy

from model import MLP
from logistic_regression import Logistic

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

class TestLogisticModel(unittest.TestCase):
    def test_binary(self):
        # Test logistic regression on a binary problem.
        iris = load_iris()
        target = (iris.target > 0).astype(np.intp)

        clf = Logistic()
        clf.fit(iris.data, target)

        self.assertEqual(clf.theta.shape, (iris.data.shape[1],))
        self.assertTrue(clf.score(iris.data, target) > 0.9)

    def test_logistic_iris(self):
        # Test logistic regression on a multi-class problem
        # using the iris dataset
        iris = load_iris()

        n_samples, n_features = iris.data.shape

        target = iris.target_names[iris.target]

        # Test that OvR (one versus rest) solvers handle
        # multiclass data correctly and give good accuracy
        # score (>0.95) for the training data.
        clf = OneVsRestClassifier(Logistic())
        clf.fit(iris.data, target)
        assert_array_equal(np.unique(target), clf.classes_)

        pred = clf.predict(iris.data)
        self.assertTrue(np.mean(pred == target) > .95)

        probabilities = clf.predict_proba(iris.data)
        assert_array_almost_equal(probabilities.sum(axis=1),
                                np.ones(n_samples))

        pred = iris.target_names[probabilities.argmax(axis=1)]
        self.assertTrue(np.mean(pred == target) > .95)

    def test_log_loss(self):
        # the loss function of LogisticRegression
        # compared with the implementation of sklearn
        n_features = 4
        X, y = make_classification(n_samples=100, n_features=n_features, random_state=0)
        lr1 = LogisticRegression(random_state=0, fit_intercept=False, C=1500)
        lr1.fit(X, y)
        clf = Logistic(max_iter=400)
        clf.fit(X, y)
        lr1_loss = _logistic_loss(lr1.coef_.reshape(n_features), X, 2 * y - 1, 0)
        clf_loss = clf.log_loss(X, y)
        self.assertTrue(np.abs(lr1_loss - clf_loss) < 1e-5)
        pass
        
if __name__=="__main__":
    unittest.main()
