### TTS技术学习研究
#### 传统技术
传统的语音合成系统分为前端和后端两个模块，前端的作用是对文本进行分析，提取后端需要的语言学信息，对于中文合成系统而言，前端一般包括文本正则化，分词，词性预测，多音字消歧，韵律预测的子模块。后端则根据前端的分析结果，通过一定的方法生成语音波形，后端系统一般分为基于统计参数建模的语音合成(或称参数合成)以及基于单元挑选和波形拼接的语音合成(或称拼接合成)，下面是两种后端系统的对比：

对于后端系统中的参数合成而言，该方法在训练阶段对语言声学特征、时长信息进行上下文相关建模，在合成阶段通过时长模型和声学模型预测声学特征参数，对声学特征参数做后处理，最终通过声码器恢复语音波形。该方法可以在语音库相对较小的情况下，得到较为稳定的合成效果。缺点在于统计建模带来的声学特征参数“过平滑”问题，以及声码器对音质的损伤。

对于后端系统中的拼接合成而言，训练阶段与参数合成基本相同，在合成阶段通过模型计算代价来指导单元挑选，采用动态规划算法选出最优单元序列，再对选出的单元进行能量规整和波形拼接。拼接合成直接使用真实的语音片段，可以最大限度保留语音音质；缺点是需要的音库一般较大，而且无法保证领域外文本的合成效果。

#### WaveNet
原文链接：

模型是基于PixelCNN重新设计的，使用了causal convolution保证预测点不会用到未来信息，预测获得的点会用作输入帮助预测新的样本。工作过程图示如下：

<div align="center">
<img src="graph/wavenet.jpg" width=500>
</div>

为了扩大感受野（receptive field）的同时不增加太多的计算量，使用了dilated convolution，简单地说就是在卷积核上打孔，抛弃一些信息，如下图所示：
<div align="center">
<img src="graph/wavenet_dialateconv.jpg" width=500>
</div>

预测输出使用了softmax分布，raw格式信号都是16bit的整数值序列，所以每个时间点的输出有65536种可能性，他们首先对数据应用μ律压扩变换，然后将其量化为256个可能的值：

<a href="https://www.codecogs.com/eqnedit.php?latex=f\left(x_{t}\right)=\operatorname{sign}\left(x_{t}\right)&space;\frac{\ln&space;\left(1&plus;\mu\left|x_{t}\right|\right)}{\ln&space;(1&plus;\mu)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f\left(x_{t}\right)=\operatorname{sign}\left(x_{t}\right)&space;\frac{\ln&space;\left(1&plus;\mu\left|x_{t}\right|\right)}{\ln&space;(1&plus;\mu)}" title="f\left(x_{t}\right)=\operatorname{sign}\left(x_{t}\right) \frac{\ln \left(1+\mu\left|x_{t}\right|\right)}{\ln (1+\mu)}" /></a>
