# Ch7 学习笔记

## Task1, 请认真阅读如下官方文档及相关代码：
- Tensorflow seq2seq
  通过这篇文章了解了seq2seq的基本轮廓，理解了针对不同长度的句子怎样使用多个bucket来封装。
  阅读文中提到的参考代码对理解很有帮助。
  [models/tutorials/rnn/translate/translate.py](https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate)   
- Variable Scope主要用于轻量级的实现variable的重用，对于RNN，encoder和decoder会使用同一组variable.
  使用不同的bucket size时也需要重用大量的variable。
- seq2seq source code
  代码结构清晰，对理解embedding_rnn_seq2seq等API的使用很有帮助。
  有时间再尝试一下embedding_attention_seq2seq等高级API

## Task2, 构建神经网络翻译模型
- 尝试了使用BasicRNNCell和GRUCell，感觉GRUCell训练速度更快，learn rate可以设得更高
- 尚未使用Bucketing，使用固定的长度，英文句长15，中文20，可以覆盖大部分语料，超长句子之间截断，最后一个词改为EOS
- 模型训练时间是个瓶颈，一开始使用全部训练样本，中英文词表都设为20000，一个epoch就要4-5小时，放弃了
- 使用小样本（300条数据）训练100个epoch，cost收敛不错，训练样本的decode结果跟label完全一致，但测试样本的翻译效果很差，过拟合明显
- 正在尝试用更大的样本训练，预期翻译效果会更好

## 学习心得
 通过这段时间的学习和实践，对深度学习和NLP有了一些粗浅体会，这是我总结的几个关键学习环节：
 1. 理解模型的基本结构和数学原理，掌握CNN，RNN，Word Embedding， Gradient descent等基本原理，对后续的学习非常有帮助，否则会知其然不知其所以然。配合阅读一些论文效果会更好，目前我看的论文数量还不多，后面有时间要多看论文。开课前读过Michael Nielsen的书[Neural Networks and Deep Learning](Neural Networks and Deep Learning), 很有帮助。
 2. 掌握tensorflow对应API的使用，重点是理解各个参数的数据维度。学习方法是看官方API文档，同时结合别人写的例子代码来看，如果对参数结构不明白的，最好运行例子代码，把每个参数的shape打印出来，会加快理解。
 3. 准备训练数据，第一步是要原始数据做预处理，然后按照模型API的要求生成训练数据集。这部分代码工作量不小，不过慢慢可以积累一些可以复用的代码，可参考tensorflow官方教程中的一些数据处理代码。
 4. 训练模型，需要尝试调整超参数集，估计训练时间，尽快实现模型收敛。对训练时间长的模型，训练结果要保存下来，以后可以继续训练或用来做预测，否则下次又要重头开始训练。
