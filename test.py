import copy

import numpy as np

from lfdnn import tensor
from lfdnn import sigmoid, add, reduce_mean, product, matmul, CE_with_logit, scale, conv2D

if __name__=="__main__":
    '''
    Here we use a very simple example to check the result
    with the gradient result given by difference limit.
    '''
    
    a = tensor([2,2], 'a')
    b = tensor([2,1], 'b')
    c = tensor([1,2], 'c')
    a2 = sigmoid(product(a, a))
    d = sigmoid(matmul(a2, b))
    d = add(d, matmul(a2, b))
    e = sigmoid(matmul(c, a2))
    c = add(matmul(e, d), scale(CE_with_logit(a2, a2), -1))
    img = tensor([2,10,10,1],'img')
    w = tensor([3,3,1,5],'weight')
    t = reduce_mean(sigmoid(conv2D(img,w)))

    feed0 = {'a':np.array([[1.,2],[3,4.5]]),'b':np.array([[1.],[2]]),'c':np.array([[1.,2]]),'img':np.random.standard_normal([2,10,10,1]),'weight':np.random.standard_normal([3,3,1,5])}

    # Test Example 1: Basic Operation
    feed = copy.deepcopy(feed0)
    print(a.back(c,feed))
    for i in range(2):
        for j in range(2):
            feed2 = copy.deepcopy(feed0)
            feed2['a'][i][j] = feed2['a'][i][j]+2e-3
            print((c.eval(feed2)-c.eval(feed))/2e-3)

    # Test Example 2: Convolution
    feed = copy.deepcopy(feed0)
    print(img.back(t,feed)[0,0,0,0])
    feed2 = copy.deepcopy(feed0)
    feed2['img'][0,0,0,0] += 2e-3
    print((t.eval(feed2)-t.eval(feed))/2e-3)
