### LSTM中的stateful
#### stateful LSTM的训练
官方文档解释：

>stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.

当stateful=True意味着在训练时每个batch的状态都会传递到下一个batch的训练中，作为下一个batch训练的初始状态。而stateful=False时，每训练完一个batch网络中的状态都会被重置。所以一个stateful的模型可以学习到不同batch之间的关联，也就是把整个数据集看作一个更大的sequence。

#### 通过一个实例的解释

> 在用LSTM做语音增强时，用传统的方法我们将连续的几帧语音信号看作不同的time steps，例如用连续的6帧语音来预测第6帧语音的一个soft mask用以去除噪声，所以在testing时我们也需要这样，但是有一个缺点就是，一段语音原本是连续的，对第一帧语音的增强似乎有一些信息也可以用在对最后一帧的语音增强上。所以在训练时，我们可以把time step设为1，batch中的每一行代表一个语音数据，这样可以保证state只在相同的语音文件数据中传递，完成整个语音数据的训练后reset state，这样做reset之前的每一次batch训练可以看成一个time step，这样做的优点是可以训练更长的time step，缺点是参数更新时，只能考虑当前帧的结果，而无法传递到之前的time step中更新参数，所以这个方法不成立。

#### stateful LSTM的测试
