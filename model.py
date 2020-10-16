import numpy as np
import lfdnn

class MLP(lfdnn.Graph):
    '''
        EpochNum: int, number of epochs in traning
        BatchSize: int, batch size used in SGD, default to all data
        InputDim: int, number of feature for each input data
        OutputDim: int, number of classes for output label
        LayerNum: int, number of intermediate layer
        HiddenNum: array-like, len(HiddenNum) = LayerNum, the number of nodes at each hidden layer
        LearningRate: double, learning rate in SGD
        _lambda: double, regularization parameter
    '''
    def __init__(self, learning_rate=0.05, epoch_num=1, batch_size='auto', hidden_layer_sizes=(), _lambda=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self._lambda = _lambda
        super().__init__(learning_rate=learning_rate, epoch_num=epoch_num, batch_size=batch_size)

    def construct_model(self, x_train, y_train):
        # get number of features
        InputDim = x_train.shape[-1]
        # get number of classes
        OutputDim = len(np.unique(y_train))
        LayerNum = len(self.hidden_layer_sizes)
        HiddenNum = self.hidden_layer_sizes
        BatchSize = self.batch_size
        _lambda = self._lambda
        if BatchSize == 'auto':
            # use all data
            BatchSize = x_train.shape[0]

        self.input = lfdnn.tensor([BatchSize, InputDim], 'Input')
        self.label = lfdnn.tensor([BatchSize, OutputDim], 'Label')
        h = self.input
        for i in range(LayerNum):
            if i == 0:
                w = lfdnn.tensor([InputDim, HiddenNum[i]], 'Weight' + str(i))
                self.weight['Weight' + str(i)] = w
            else:
                w = lfdnn.tensor([HiddenNum[i - 1], HiddenNum[i]], 'Weight' + str(i))
                self.weight['Weight' + str(i)] = w
            b = lfdnn.tensor([1, HiddenNum[i]],'Bias' + str(i))
            self.weight['Bias' + str(i)] = b
            h = lfdnn.add(lfdnn.matmul(h, w), b)
            h = lfdnn.sigmoid(h)
        if len(HiddenNum) > 0:
            w = lfdnn.tensor([HiddenNum[-1], OutputDim], 'output_weight')
        else:
            w = lfdnn.tensor([InputDim, OutputDim], 'output_weight')
        self.weight['output_weight'] = w
        b = lfdnn.tensor([1, OutputDim], 'output_bias')
        self.weight['output_bias'] = b
        h = lfdnn.add(lfdnn.matmul(h, w), b)
        self.output = lfdnn.softmax(h)
        self.loss = lfdnn.CE_with_logit(h, self.label)
        if _lambda > 0:
            for w in self.weight.values():
                self.loss = lfdnn.add(self.loss, lfdnn.scale(lfdnn.reduce_mean(lfdnn.product(w, w)), _lambda))
        self.accuracy = lfdnn.accuracy(self.output, self.label)
