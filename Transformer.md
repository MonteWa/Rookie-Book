## Transformer
1. Transformer 就是 seq2seq with self-attention.
### Transformer 的产生
1. RNN不能进行平行处理，所有有人提出使用CNN来代替RNN，例如用一个filter来连续处理序列输入，但是一个filter只受一个时间点的影响，失去了RNN可以利用上下文的优势。
2. 但是对输入序列再使用filter处理前一层输出的多个输入，这样可以扩大高层filter的感受野，这样就可以考虑到上下文。
3. 虽然使用CNN可以平行处理，但是需要的层数比较深，导致网络结构复杂。于是用self-attention来取代RNN可以做的事，self-attention可以被认为事一个layer。所以self-attention的神奇之处在于，它既可以考虑上下文，又可以平行计算。
<div align="center">
<img src="graph/transformer_rnn.jpg" width=400>
</div>

4. self-attention layer: 得到q,k,v后，用每一个q对每一个k做attention（attention就是输出两个向量的匹配程度），经过一番操作，最后的输出是包含整句信息的，并且可以平行运算。
5. 但是self-attention的输出没有考虑上下文的具体位置信息，所以还要在被qkv处理之前要添加额外的位置信息。
