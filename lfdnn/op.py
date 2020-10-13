from lfdnn.main import tensor,NameManager
import numpy as np

NM = NameManager()

def matmul(x1,x2):
    out = tensor(x1.shape[:-1]+x2.shape[1:],NM.get('matmul'),'matmul',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def add(x1,x2):
    out = tensor(x1.shape,NM.get('add'),'add',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def sigmoid(x):
    out = tensor(x.shape,NM.get('sigmoid'),'sigmoid',[x])
    x.output_list.append(out)
    return out

def relu(x):
    out = tensor(x.shape,NM.get('relu'),'relu',[x])
    x.output_list.append(out)
    return out

def log(x):
    out = tensor(x.shape,NM.get('log'),'log',[x])
    x.output_list.append(out)
    return out

def product(x1,x2):
    out = tensor(x1.shape,NM.get('product'),'product',[x1,x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def softmax(x):
    out = tensor(x.shape,NM.get('softmax'),'softmax',[x])
    x.output_list.append(out)
    return out

def log_softmax(x):
    # We found that log(softmax(x)) have serious numerical issue.
    # Therefore we command use log_softmax() instead.
    out = tensor(x.shape,NM.get('log_softmax'),'log_softmax',[x])
    x.output_list.append(out)
    return out

def reduce_sum(x):
    out = tensor([1,1],NM.get('reduce_sum'),'reduce_sum',[x])
    x.output_list.append(out)
    return out

def scale(x,alpha):
    out = tensor(x.shape,NM.get('scale'),'scale',[x,alpha])
    x.output_list.append(out)
    return out

def reduce_mean(x):
    out = scale(reduce_sum(x),1/np.product(x.shape))
    return out

def CE(x,y):
    out = scale(reduce_mean(product(y,log(x))),-1)
    return out

def CE_with_logit(x,y):
    out = scale(reduce_mean(product(y,log_softmax(x))),-1)
    return out

def accuracy(pred,y):
    out = tensor([1,1],NM.get('accuracy'),'accuracy',[pred,y])
    pred.output_list.append(out)
    y.output_list.append(out)
    return out

def imagePadding(x,pad_size=1):
    new_shape = [int(x.shape[i]+[0,pad_size*2,pad_size*2,0][i]) for i in range(len(x.shape))]
    out = tensor(new_shape,NM.get('imagePadding'),'imagePadding',[x,int(pad_size)])
    x.output_list.append(out)
    return out

def imageZIP(x,kernel_size=3,stride=1):
    x_pad = imagePadding(x,(kernel_size-1)/2)
    new_shape = [int(x.shape[0]*(x.shape[1]/stride)*(x.shape[2]/stride)),
        int(kernel_size*kernel_size*x.shape[3])]
    out = tensor(new_shape,NM.get('imageZIP'),'imageZIP',[x_pad,kernel_size,stride])
    x_pad.output_list.append(out)
    return out

def reshape(x,target_shape):
    out = tensor(target_shape,NM.get('reshape'),'reshape',[x,target_shape])
    x.output_list.append(out)
    return out

def conv2D(x,w,stride=1):
    kernel_size = w.shape[0]
    x_ZIP = imageZIP(x,kernel_size,stride)
    w_ZIP = reshape(w, [-1,w.shape[-1]])
    out = matmul(x_ZIP,w_ZIP)
    out_img = reshape(out, x.shape[:-1]+[w.shape[-1]])
    return out_img 


if __name__=="__main__":
    '''
    Here we use a very simple example to check the result
    with the gradient result given by difference limit.
    '''
    import copy
    a = tensor([2,2],'a')
    b = tensor([2,1],'b')
    c = tensor([1,2],'c')
    a2 = sigmoid(product(a,a))
    d = sigmoid(matmul(a2,b))
    d = add(d,matmul(a2,b))
    e = sigmoid(matmul(c,a2))
    c = add(matmul(e,d),scale(CE_with_logit(a2,a2),-1))
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
