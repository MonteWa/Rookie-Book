### LSTM中的stateful
#### stateful LSTM的训练
官方文档解释：

>stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.

当stateful=True意味着在训练时每个batch的状态都会传递到下一个batch的训练中，作为下一个batch训练的初始状态。而stateful=False时，每训练完一个batch网络中的状态都会被重置。所以一个stateful的模型可以学习到不同batch之间的关联，也就是把整个数据集看作一个更大的sequence。

#### 通过一个实例的解释



#### stateful LSTM的测试
