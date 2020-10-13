import unittest
import copy

import numpy as np

from lfdnn import tensor
from lfdnn import sigmoid, add, reduce_mean, product, matmul, CE_with_logit, scale, conv2D

class TestLibrary(unittest.TestCase):
    def test_shape(self):
        a = tensor([1, 2], 'a')
        self.assertEqual(reduce_mean(a).shape, [1, 1])
    def test_forward(self):
        a = tensor([1, 2], 'a')
        feed = {'a': np.array([[5, 6]])}
        self.assertAlmostEqual(reduce_mean(a).forward(feed), 5.5)

if __name__=="__main__":
    unittest.main()
