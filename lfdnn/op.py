from lfdnn.main import tensor,NameManager
import numpy as np

NM = NameManager()

def matmul(x1, x2):
    out = tensor(x1.shape[:-1]+x2.shape[1:],NM.get('matmul'), 'matmul', [x1, x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def add(x1, x2):
    out = tensor(x1.shape, NM.get('add'), 'add', [x1, x2])
    x1.output_list.append(out)
    if x1 is not x2:
        x2.output_list.append(out)
    return out

def sigmoid(x):
    out = tensor(x.shape,NM.get('sigmoid'),'sigmoid',[x])
    x.output_list.append(out)
    return out

def relu(x):
    out = tensor(x.shape, NM.get('relu'), 'relu', [x])
    x.output_list.append(out)
    return out

def log(x):
    out = tensor(x.shape, NM.get('log'), 'log', [x])
    x.output_list.append(out)
    return out

def product(x1, x2):
    '''
    elementwise multiplication of two tensors
    '''
    out = tensor(x1.shape, NM.get('product'), 'product', [x1, x2])
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
    out = tensor([1,1], NM.get('reduce_sum'), 'reduce_sum', [x])
    x.output_list.append(out)
    return out

def scale(x,alpha):
    out = tensor(x.shape,NM.get('scale'),'scale',[x,alpha])
    x.output_list.append(out)
    return out

def reduce_mean(x):
    out = scale(reduce_sum(x), 1/np.product(x.shape))
    return out

def CE(x,y):
    out = scale(reduce_mean(product(y,log(x))), -1)
    return out

def CE_with_logit(x,y):
    out = scale(reduce_mean(product(y, log_softmax(x))), -1)
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


