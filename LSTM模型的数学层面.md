### RNN和LSTM模型的数学层面

参考文献：
[Understanding LSTM Networks]
[为什么相比于RNN，LSTM在梯度消失上表现更好?]
[RNN以及LSTM的介绍和公式梳理]

[Understanding LSTM Networks]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

[为什么相比于RNN，LSTM在梯度消失上表现更好?]: https://www.zhihu.com/question/44895610/answer/616818627

[RNN以及LSTM的介绍和公式梳理]: https://blog.csdn.net/dark_scope/article/details/47056361

#### 从RNN说起

先来看看RNN的前向传播：

<div align="center">
<img src="graph/RNN-unrolled.jpg" width=400>
</div>

再看看前向传播公式：

<a href="https://www.codecogs.com/eqnedit.php?latex=h_{t}=\sigma&space;\left&space;(&space;W_{xh}x_{t}&space;&plus;&space;W_{hh}h_{t-1}&plus;b_{h}\right)&space;\tag{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{t}=\sigma&space;\left&space;(&space;W_{xh}x_{t}&space;&plus;&space;W_{hh}h_{t-1}&plus;b_{h}\right)&space;\tag{1}" title="h_{t}=\sigma \left ( W_{xh}x_{t} + W_{hh}h_{t-1}+b_{h}\right) \tag{1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=y_{t}=W_{hy}&plus;b_{y}&space;\tag{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{t}=W_{hy}&plus;b_{y}&space;\tag{2}" title="y_{t}=W_{hy}+b_{y} \tag{2}" /></a>

公式(1)中 <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a>是隐藏层的激活函数，一般为Sigmoid function，输入是一个序列  <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;\left&space;\{x_{t}...x_{T}&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;\left&space;\{x_{t}...x_{T}&space;\right&space;\}" title="x = \left \{x_{t}...x_{T} \right \}" /></a>, 输出就是迭代的计算 <a href="https://www.codecogs.com/eqnedit.php?latex=h_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{t}" title="h_{t}" /></a> 和 <a href="https://www.codecogs.com/eqnedit.php?latex=y_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{t}" title="y_{t}" /></a>。

#### RNN面临的问题
RNN主要面临的是在处理长序列的问题时，梯度消失的问题。[Understanding LSTM Networks]中的对问题的描述如下：
>But there are also cases where we need more context. Consider trying to predict the last word in the text “I grew up in France… I speak fluent French.” Recent information suggests that the next word is probably the name of a language, but if we want to narrow down which language, we need the context of France, from further back. It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.

当上下文相关信息在time step上的距离越来越远时，RNN就很难利用到前面的上下文信息了。


这个问题主要原因在于公式（1）这一项：


<a href="https://www.codecogs.com/eqnedit.php?latex=h_{t}=\sigma&space;\left&space;(&space;W_{xh}x_{t}&space;&plus;&space;W_{hh}h_{t-1}&plus;b_{h}\right)&space;\tag{1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{t}=\sigma&space;\left&space;(&space;W_{xh}x_{t}&space;&plus;&space;W_{hh}h_{t-1}&plus;b_{h}\right)&space;\tag{1}" title="h_{t}=\sigma \left ( W_{xh}x_{t} + W_{hh}h_{t-1}+b_{h}\right) \tag{1}" /></a>

我们可以想象 $h_{t-1}$ 里其实是包含 $h_{0}$ 到 $h_{t-2}$ 这些项的，所以我们在进行反向传播时，在求 $W_{hh}$ 的梯度时根据链式法则会遇到一个很长的 $h_{t}$ 的梯度的导数 $h_{t}^{'}$ 的连乘。就像下面这样：

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L_{t}}{\partial&space;W^{h}}=\sum_{t=0}^{T}&space;\sum_{k=0}^{t}&space;\frac{\partial&space;L_{t}}{\partial&space;y_{t}}&space;\frac{\partial&space;y_{t}}{\partial&space;h_{t}}\left(\prod_{j=k&plus;1}^{t}&space;\frac{\partial&space;h_{j}}{\partial&space;h_{j-1}}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L_{t}}{\partial&space;W^{h}}=\sum_{t=0}^{T}&space;\sum_{k=0}^{t}&space;\frac{\partial&space;L_{t}}{\partial&space;y_{t}}&space;\frac{\partial&space;y_{t}}{\partial&space;h_{t}}\left(\prod_{j=k&plus;1}^{t}&space;\frac{\partial&space;h_{j}}{\partial&space;h_{j-1}}\right)" title="\frac{\partial L_{t}}{\partial W^{h}}=\sum_{t=0}^{T} \sum_{k=0}^{t} \frac{\partial L_{t}}{\partial y_{t}} \frac{\partial y_{t}}{\partial h_{t}}\left(\prod_{j=k+1}^{t} \frac{\partial h_{j}}{\partial h_{j-1}}\right)" /></a>

从(1)中我们可得：

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;h_{t}}{\partial&space;h_{t-1}&space;}&space;=&space;\sigma&space;^{'}W_{hh}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;h_{t}}{\partial&space;h_{t-1}&space;}&space;=&space;\sigma&space;^{'}W_{hh}" title="\frac{\partial h_{t}}{\partial h_{t-1} } = \sigma ^{'}W_{hh}" /></a>

而其中sigmoid的导数为：

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma&space;^{'}=\sigma\left&space;(1-\sigma&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma&space;^{'}=\sigma\left&space;(1-\sigma&space;\right&space;)" title="\sigma ^{'}=\sigma\left (1-\sigma \right )" /></a>

所以 $\sigma ^{'}$ 的大小在0到0.25之间，可以想象这样一个因子在梯度计算中被重复相乘梯度会越来越小，导致两个在time step距离较远的输入无法产生足够的影响。

#### 再看看LSTM
LSTM前向传播的图示如下：

<div align="center">
<img src="graph/lstm-lhy.jpg" width=400>
</div>

LSTM的前向传播公式如下：

<a href="https://www.codecogs.com/eqnedit.php?latex=i_{t}&space;=\sigma\left(W_{x&space;i}&space;x_{t}&plus;W_{h&space;i}&space;h_{t-1}&plus;W_{c&space;i}&space;c_{t-1}&plus;b_{i}\right" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i_{t}&space;=\sigma\left(W_{x&space;i}&space;x_{t}&plus;W_{h&space;i}&space;h_{t-1}&plus;W_{c&space;i}&space;c_{t-1}&plus;b_{i}\right" title="i_{t} =\sigma\left(W_{x i} x_{t}+W_{h i} h_{t-1}+W_{c i} c_{t-1}+b_{i}\right" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{t}&space;=\sigma\left(W_{x&space;f}&space;x_{t}&plus;W_{h&space;f}&space;h_{t-1}&plus;W_{c&space;f}&space;c_{t-1}&plus;b_{f}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{t}&space;=\sigma\left(W_{x&space;f}&space;x_{t}&plus;W_{h&space;f}&space;h_{t-1}&plus;W_{c&space;f}&space;c_{t-1}&plus;b_{f}\right)" title="f_{t} =\sigma\left(W_{x f} x_{t}+W_{h f} h_{t-1}+W_{c f} c_{t-1}+b_{f}\right)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=c_{t}&space;=f_{t}&space;c_{t-1}&plus;i_{t}&space;\tanh&space;\left(W_{x&space;c}&space;x_{t}&plus;W_{h&space;c}&space;h_{t-1}&plus;b_{c}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{t}&space;=f_{t}&space;c_{t-1}&plus;i_{t}&space;\tanh&space;\left(W_{x&space;c}&space;x_{t}&plus;W_{h&space;c}&space;h_{t-1}&plus;b_{c}\right)" title="c_{t} =f_{t} c_{t-1}+i_{t} \tanh \left(W_{x c} x_{t}+W_{h c} h_{t-1}+b_{c}\right)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=o_{t}&space;=\sigma\left(W_{x&space;o}&space;x_{t}&plus;W_{h&space;o}&space;h_{t-1}&plus;W_{c&space;o}&space;c_{t}&plus;b_{o}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?o_{t}&space;=\sigma\left(W_{x&space;o}&space;x_{t}&plus;W_{h&space;o}&space;h_{t-1}&plus;W_{c&space;o}&space;c_{t}&plus;b_{o}\right)" title="o_{t} =\sigma\left(W_{x o} x_{t}+W_{h o} h_{t-1}+W_{c o} c_{t}+b_{o}\right)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=h_{t}&space;=o_{t}&space;\tanh&space;\left(c_{t}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{t}&space;=o_{t}&space;\tanh&space;\left(c_{t}\right)" title="h_{t} =o_{t} \tanh \left(c_{t}\right)" /></a>

需要注意几点细节：
  * 首先input gate和foget gate不仅由当前时间点的输入<a href="https://www.codecogs.com/eqnedit.php?latex=x_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{t}" title="x_{t}" /></a> 控制，还受到上个时间点记忆<a href="https://www.codecogs.com/eqnedit.php?latex=c_{t-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{t-1}" title="c_{t-1}" /></a> 和上个时间点的输出<a href="https://www.codecogs.com/eqnedit.php?latex=h_{t-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{t-1}" title="h_{t-1}" /></a> 共同控制（看第一行和第二行）。
  * 影响当前即已更新的输入是当前时间点输入<a href="https://www.codecogs.com/eqnedit.php?latex=x_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{t}" title="x_{t}" /></a> 和上一个时间点的输出<a href="https://www.codecogs.com/eqnedit.php?latex=h_{t-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{t-1}" title="h_{t-1}" /></a>。
  * 控制output gate与其他两个gate不同的是输入中不包含上个时间点记忆<a href="https://www.codecogs.com/eqnedit.php?latex=c_{t-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{t-1}" title="c_{t-1}" /></a> 取而代之的是当前更新过的记忆<a href="https://www.codecogs.com/eqnedit.php?latex=c_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{t}" title="c_{t}" /></a> 。

#### LSTM是怎么解决RNN的问题的呢？
LSTM缓解了梯度消失的问题，那么他是如何做到的呢？RNN中的主要问题出在$h_{t}$ 再做反向传播时梯度串联的相乘，越来越小的问题上，导致较早期的记忆无法对较远的判断造成影响。而在LSTM模型中，我们看看记忆这一项是怎么传播的:

<a href="https://www.codecogs.com/eqnedit.php?latex=c_{t}&space;=f_{t}&space;c_{t-1}&plus;i_{t}&space;\tanh&space;\left(W_{x&space;c}&space;x_{t}&plus;W_{h&space;c}&space;h_{t_1}&plus;b_{c}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{t}&space;=f_{t}&space;c_{t-1}&plus;i_{t}&space;\tanh&space;\left(W_{x&space;c}&space;x_{t}&plus;W_{h&space;c}&space;h_{t_1}&plus;b_{c}\right)" title="c_{t} =f_{t} c_{t-1}+i_{t} \tanh \left(W_{x c} x_{t}+W_{h c} h_{t_1}+b_{c}\right)" /></a>

这里再反向传播时：

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;c_{t}}{\partial&space;c_{t-1}}=f_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;c_{t}}{\partial&space;c_{t-1}}=f_{t}" title="\frac{\partial c_{t}}{\partial c_{t-1}}=f_{t}" /></a>

而：

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{t}&space;=\sigma\left(x\right)&space;\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{t}&space;=\sigma\left(x\right)&space;\\" title="f_{t} =\sigma\left(x\right) \\" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a> 是一个sigmoid函数，它上限为1下线为0，也就是说模型只需要决定门的开关就行了，主要收到深度影响的其实是对门限的更新能力，所以相比于RNN来说，LSTM能够更好的传递上下文的信息。

#### 总结
