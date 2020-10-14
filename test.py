import unittest
import copy

import numpy as np

from lfdnn import tensor
from lfdnn import sigmoid, add, reduce_mean, product, matmul, CE_with_logit, scale

from lfdnn.utils import TensorOpUndefinedError, TensorOpNotSupported

class TestLibrary(unittest.TestCase):
    def test_shape(self):
        a = tensor([1, 2], 'a')
        self.assertEqual(reduce_mean(a).shape, [1, 1])
    def test_forward(self):
        a = tensor([1, 2], 'a')
        feed = {'a': np.array([[5, 6]])}
        self.assertAlmostEqual(reduce_mean(a).forward(feed), 5.5)
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

if __name__=="__main__":
    unittest.main()
