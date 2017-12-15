## ch6 RNN

### Lectures

* softmax 与交叉熵
* softmax 性能优化
* 循环神经网络
* 变长数据处理

### Tasks

* 1 - 不同 batch_size 模型预测所需时间对比
  * 过程: 见 [ch4/nn_language_model.ipynb](../ch4/nn_language_model.ipynb) 最后
  * 结果: 
    * 预测的 batch_size 从 1 增大到 100, 耗费的时间竟然没怎么增加
    * 从 1 到 1000, 耗费的时间也没有超过两倍.
* 2 - 改进 ch4 的神经网络语言模型性能(速度)
  * 用 nce loss 改进, 见 [nn_lang_model_nce.ipynb](nn_lang_model_nce.ipynb)
  * 结果: 训练时间由约 450 s 缩短到约 100 s, 效果明显
* 3 - 使用 RNN 构建语言模型
  * 见 [RNN_language_model.ipynb](RNN_language_model.ipynb)

