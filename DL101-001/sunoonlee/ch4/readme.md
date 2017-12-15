## ch4 神经网络语言模型

### lectures

* 更深入地理解神经网络
* 神经网络的表达能力
* 矩阵符号表示神经网络
* 神经网络语言模型

### tasks

- 1、基于矩阵乘法用 Tensorflow 实现 ch3 作业的单隐层神经网络
- 2、在 tf.matrix.ipynb 中，将列向量表示形式，改成行向量表示形式。进一步，使用 tf.layers.dense 替换原有的矩阵乘法。
  - 见 [simple_nn_ch4.ipynb](simple_nn_ch4.ipynb)
- 2a、在词向量神经网络中，（当词向量为行向量时）为什么取 weight matrix 的一行，与做矩阵乘法效果一样？
  - 动手写一下就知道了
```
                1 2
[ 0 0 1 0 ] x [ 3 4 ] = [ 5 6 ]
                5 6
                7 8
```
- 3、构建 word embedding
  - 见 [nn_language_model.ipynb](nn_language_model.ipynb)
  - [cs224n lec2 note: word vector](lec2-word-vectors.md)
