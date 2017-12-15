## ch5: CNNs

### Lectures

* softmax
* Tensorboard
* CNNs 用于图像处理
* CNNs 用于 NLP


### Tasks

* 1 - 神经网络用于情感分类
    - 稀疏特征 + 词向量 + 隐层 + Softmax
    - 见 [nn_sentiment.ipynb](nn_sentiment.ipynb)
* 2 - Tensorboard 数据可视化
    - 截图见 [ch4/tf_summary_cost.png](../ch4/tf_summary_cost.png), [ch4/tf_summary_embeddings.png](../ch4/tf_summary_embeddings.png)
    - 顺便更新了 ch4 task3 [ch4/nn_language_model.ipynb](../ch4/nn_language_model.ipynb), 结果比之前有明显提升
* 3a - 利用卷积平滑图像
    - 见 [conv_image.ipynb](conv_image.ipynb)
* 3b - 卷积神经网络实现情感分类
    - notebook 见 [conv_nlp.ipynb](conv_nlp.ipynb)
    - script 见 [conv_nlp.py](conv_nlp.ipynb) 及 [data_reader.py](data_reader.py) 
        + Usage: `python conv_nlp.py train_shuffle.txt test_shuffle.txt`
