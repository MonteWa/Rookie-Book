### Batch Normalization的作用和原理
#### 从实用性出发
使用batch normalization的目的是加速训练，keras中对Batchnormalization的解释为：

>Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

标准化前一层的激活输出，将其变换到均值接近0，标准差接近1的分布范围上。

Batch normalization的论文对该算法的描述如下：[Batch Normalization]

  [Batch Normalization]: https://arxiv.org/pdf/1502.03167.pdf

<div align="center">
<img src="batch_norm.png" width=60%>
</div>

从算法中我们可以看出，上一层的激活值首先会被统计求出均值 $\mu$ 和方差 $\sigma^{^{2}}$, 然后进行标准化，注意这里标准化时方差需要加上极小值 $\epsilon$，目的是防止除数为0.

注意最后输出的:
$$ y = \gamma \cdot x + \beta $$
这里 $\gamma$和 $\beta$ 是两个可以更新的参数，这样做的原因原论文的结束如下：

>Note that simply normalizing each input of a layer may
change what the layer can represent. For instance, normalizing
the inputs of a sigmoid would constrain them to
the linear regime of the nonlinearity. To address this, we
make sure that the transformation inserted in the network
can represent the identity transform.

也就是说，如果只是简单的将输入标准话，有可能会改变它原本想要表达的内容，例如标准话sigmoid函数的输入会使他们强行落在线性范围内，为了解决这一问题而引入的这两个可以被训练的参数。

#### 它解决了什么问题？
原文中关于这一问题的表述如下：
>Towards Reducing Internal Covariate Shift

为了减少内部协变量偏移。

>We define Internal Covariate Shift as the change in the
distribution of network activations due to the change in
network parameters during training.

在训练过程中，网络参数的变化会导致激活结果分布的改变，我们都知道在输入数据前对数据做标准化可以加速训练，那么如果能够对每一个激活层都做标准化就可以进一步的加速训练。

#### 在训练过程中Batch normalization层发生了什么？
原文中的反向转播梯度更新算法如下：

<div align="center">
<img src="batch_norm_update.png" width=60%>
</div>

其实可以将batch normalization看作一种特殊的激活方式，输入 $x$ 被标准化函数激活，所以标准化函数的梯度会引入计算，上式中第四行就代表的是loss对被标准化之前的 $x$的求导。

#### 在激活之前白化还是激活之后白化？
这个问题似乎没有标准答案，正如前面所讨论的BN就是将每层数据标准化，默认的做法是将在激活之后进行BN操作，但是如果在数据进入激活函数之前进行标准化，这样一来那么应该有近一半的激活输出为0，。。。

所以还是都实验一下比较好。

#### 通过一个实验对比
