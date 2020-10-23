import unittest
import copy
import sys

import numpy as np

from sklearn.datasets import load_iris, make_classification
from sklearn.multiclass import OneVsRestClassifier
# scikit-learn >= 0.23 has this private API `_logistic_loss`
from sklearn.linear_model._logistic import _logistic_loss
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.datasets import load_digits

import lfdnn
from lfdnn import tensor
from lfdnn import operator
from lfdnn.numerical_tensor import numerical_accuracy

from model import MLP
from logistic_regression import Logistic
from ridge_regression import RidgeRegression

class TestAutoDifferential(unittest.TestCase):
    def test_shape(self):
        a = tensor([1, 2], 'a')
        self.assertEqual(operator.reduce_mean(a).shape, [1, 1])

    def test_forward(self):
        a = tensor([2], 'a')
        feed = {'a': np.array([5, 6])}
        self.assertAlmostEqual(operator.reduce_mean(a).forward(feed), 5.5)

    def test_product(self):
        a = tensor([2, 1], 'a')
        feed = {'a': np.array([[5], [6]])}
        assert_array_almost_equal(operator.product(a, a).eval(feed), np.array([[25], [36]]))

    def test_backward(self):
        a = tensor([1, 1], 'a')
        feed = {'a': np.array([[0]])}
        self.assertAlmostEqual(operator.sigmoid(a).differentiate(a, feed)[0, 0], 0.25)

    def test_backward_add(self):
        a = tensor([1, 1], 'a')
        feed = {'a': np.array([[0.1]])}
        target = operator.add(operator.product(a, a), operator.scale(a, 3))
        self.assertAlmostEqual(target.eval(feed)[0, 0], 0.31)
        self.assertAlmostEqual(operator.reduce_sum(target).differentiate(a, feed)[0, 0], 3.2)

    def test_relu_derivative(self):
        a = tensor([1, 3], 'a')
        feed = {'a': np.array([[-1, 0, 3]])}
        assert_array_almost_equal(operator.relu(a).differentiate(a, feed), np.array([[0, 0, 1]]))
        result_1 = operator.relu(operator.sigmoid(a)).differentiate(a, feed)
        result_2 = operator.sigmoid(a).differentiate(a, feed)
        assert_array_almost_equal(result_1, result_2)

    def test_numerical_accuracy(self):
        prob_vector = np.array([[0.3, 0.7], [0.6, 0.4]])
        true_vector = [1, 1]
        self.assertAlmostEqual(numerical_accuracy(prob_vector, true_vector), 0.5)
        true_vector = [1, 0]
        self.assertAlmostEqual(numerical_accuracy(prob_vector, true_vector), 1.0)

    def test_softmax(self):
        a = tensor([1, 3], 'a')
        feed = {'a': np.array([[1, 2, 3]])}
        answer_list = np.exp([1, 2, 3])
        answer_list /= np.sum(answer_list)
        assert_array_almost_equal(operator.softmax(a).forward(feed)[0], answer_list)

    def test_log_softmax(self):
        a = tensor([1, 3], 'a')
        feed = {'a': np.array([[1, 2, 3]])}
        assert_array_almost_equal(operator.log_softmax(a).forward(feed), operator.log(operator.softmax(a)).forward(feed))

    def test_relu(self):
        a = tensor([1, 3], 'a')
        feed = {'a': np.array([[1, -1, 3]])}
        assert_array_almost_equal(operator.relu(a).forward(feed), np.array([[1, 0, 3]]))

    def test_cross_entropy(self):
        x = tensor([3, 3], 'x')
        y = tensor([3, 3], 'y')
        feed = {'x': np.array([[0.4, 0.5, 0.1], [0.4, 0.5, 0.1], [0.4, 0.5, 0.1]]), 'y': np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]])}
        true_value = (2 * np.log(0.1) + np.log(0.4)) / 3
        self.assertAlmostEqual(operator.CE(x, y).eval(feed), -1.0 * true_value)

    def test_matrix_multiplication(self):
        a = tensor([2, 3], 'a')
        b = tensor([3, 1], 'b')
        feed = {'a': np.array([[0.4, 0.5, 1.1], [0.1, 2.3, -0.3]]),
                'b': np.array([[1.2], [-2.3], [0.2]])}
        true_matrix = np.array([[-0.45], [-5.23]])
        assert_array_almost_equal(operator.matmul(a, b).eval(feed), true_matrix)

    def test_mse(self):
        a = tensor([3, 1], 'a')
        b = tensor([3, 1], 'b')
        feed = {'a': np.array([[1.3], [-2.2], [0.4]]),
                'b': np.array([[1.2], [-2.3], [0.2]])}
        true_value = mean_squared_error(feed['a'], feed['b'])
        self.assertAlmostEqual(operator.mse(a, b).eval(feed), true_value)

class TestMLP(unittest.TestCase):
    def test_construction_model(self):
        mlp = MLP(hidden_layer_sizes=(2,))
        # number of data = 3
        # number of feature = 2
        x_train = np.zeros([3, 2])
        y_train = [1, 0, 1]
        mlp.train(x_train, y_train)
        self.assertTrue(mlp.input.output_list[0].output_list[0].output_list[0].name.find('softmax') < 0)

    def test_multiple_layer_with_regulation(self):
        # test on UCI ML hand-written digits datasets
        mlp = MLP(hidden_layer_sizes=(30,), epoch_num=600, batch_size=32, learning_rate=0.2, _lambda=0.05)
        digits = load_digits()
        n_samples = len(digits.images)
        x_train = digits.data[:n_samples // 2]
        y_train = digits.target[:n_samples // 2]
        np.random.seed(2020)
        mlp.train(x_train, y_train)
        self.assertTrue(numerical_accuracy(mlp.predict(x_train), y_train) > 0.99)
        x_test = digits.data[n_samples // 2:]
        y_test = digits.target[n_samples // 2:]
        self.assertTrue(numerical_accuracy(mlp.predict(x_test), y_test) > 0.93)

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

    def test_xor(self):
        X = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.float32).reshape(4,2)
        Y = np.array([0, 0, 1, 1], dtype=np.float32)
        mlp = MLP(hidden_layer_sizes=(2,), epoch_num=1600, learning_rate=0.22)
        np.random.seed(2020)
        mlp.train(X, Y)
        self.assertAlmostEqual(numerical_accuracy(mlp.predict(X), Y), 1.0)

@unittest.skipIf(RidgeRegression().skip, 'skip bonus question')
class TestRidgeModel(unittest.TestCase):
    def test_ridge(self):
        # Ridge regression convergence test
        # compare to the implementation of sklearn
        rng = np.random.RandomState(0)
        alpha = 1.0

        # With more samples than features
        n_samples, n_features = 6, 5
        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)

        ridge = Ridge(alpha=alpha, fit_intercept=True, solver='sag')
        custom_implemented_ridge = RidgeRegression(alpha=alpha)
        ridge.fit(X, y)
        np.random.seed(2020)
        custom_implemented_ridge.fit(X, y)
        self.assertEqual(custom_implemented_ridge.theta.shape, (X.shape[1], 1))
        self.assertTrue(custom_implemented_ridge.score(X, y) > ridge.score(X, y) - 0.1)

    def test_ridge_singular(self):
        # test on a singular matrix
        rng = np.random.RandomState(0)
        n_samples, n_features = 6, 6
        y = rng.randn(n_samples // 2)
        y = np.concatenate((y, y))
        X = rng.randn(n_samples // 2, n_features)
        X = np.concatenate((X, X), axis=0)

        ridge = RidgeRegression(alpha=0)
        np.random.seed(2020)
        ridge.train(X, y)
        self.assertGreater(ridge.score(X, y), 0.9)

    def test_ridge_vs_lstsq(self):
        # On alpha=0.,
        # Ridge and ordinary linear regression should yield nearly same solution.
        rng = np.random.RandomState(0)
        # we need more samples than features
        n_samples, n_features = 6, 4
        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)

        ridge = RidgeRegression(alpha=0, epoch_num=600, learning_rate=0.1)
        ols = LinearRegression(fit_intercept=True)
        np.random.seed(2020)
        ridge.fit(X, y)
        ols.fit(X, y)
        self.assertTrue(np.linalg.norm(ridge.theta.reshape([4]) - ols.coef_) < 0.01)

class TestLogisticModel(unittest.TestCase):
    def test_binary(self):
        # Test logistic regression on a binary problem.
        iris = load_iris()
        target = (iris.target > 0).astype(np.intp)

        clf = Logistic()
        np.random.seed(2010)
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
        clf = OneVsRestClassifier(Logistic(max_iter=400, learning_rate=0.15))
        np.random.seed(0)
        clf.fit(iris.data, target)
        assert_array_equal(np.unique(target), clf.classes_)

        pred = clf.predict(iris.data)
        self.assertTrue(np.mean(pred == target) > .93)

        probabilities = clf.predict_proba(iris.data)
        assert_array_almost_equal(probabilities.sum(axis=1),
                                np.ones(n_samples))

        pred = iris.target_names[probabilities.argmax(axis=1)]
        self.assertTrue(np.mean(pred == target) > .93)

    def test_log_loss(self):
        # the loss function of LogisticRegression
        # compared with the implementation of sklearn
        n_features = 4
        X, y = make_classification(n_samples=100, n_features=n_features, n_informative=4, n_redundant=0, class_sep=2.0, random_state=0)
        lr1 = LogisticRegression(random_state=0, fit_intercept=True, C=1500, max_iter=100)
        lr1.fit(X, y)
        clf = Logistic(max_iter=400, learning_rate=0.13)
        np.random.seed(2020)
        clf.fit(X, y)

        w_b_vector = lr1.coef_.reshape(n_features)
        w_b_vector = np.hstack((w_b_vector, lr1.intercept_))
        lr1_loss = _logistic_loss(w_b_vector, X, 2 * y - 1, 0)
        clf_loss = clf.log_loss(X, y)
        self.assertTrue(np.abs(lr1_loss - clf_loss) < 0.4)
        pass
        
if __name__=="__main__":
    if len(sys.argv) > 1:
        unittest.main()
    test_obj = unittest.main(exit=False)
    q1 = 5
    q2 = 5
    q3 = 0
    if len(test_obj.result.skipped) == 0:
        q3 = 2.5
    f_or_e = test_obj.result.failures
    f_or_e.extend(test_obj.result.errors)
    for failure in f_or_e:
        if str(failure[0]).find('AutoDifferential') > 0 and q1 > 0:
            q1 -= 1
        elif str(failure[0]).find('MLP') > 0 and q2 > 0:
            q2 -= 1
        elif str(failure[0]).find('LogisticModel') > 0 and q2 > 0:
            q2 -= 1
        elif str(failure[0]).find('RidgeModel') > 0 and q3 > 0:
            q3 -= 1
            if q3 < 0:
                q3 = 0
    print("Your final score of PA2: ", q1 + q2 + q3)
    if len(f_or_e) > 0:
        exit(-1)