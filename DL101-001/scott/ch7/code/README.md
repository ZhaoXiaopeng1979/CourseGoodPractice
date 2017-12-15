~ 主要用于存放本周视频中的示例代码 
- seq2seq.ipynb: 课程中相关 seq2seq 模型示例代码
- `https://github.com/demon386/tf_bucket_seq2seq` 一个基于 bucket 模型的实现（有关 bucket，可以阅读 seq2seq tutorial 中相关描述）
    - 也可直接基于这个模型或改造来完成作业

数据：
- TED 中文/英文平行语料。取自 [http://www.bfsu-corpus.org/channels/corpus](http://www.bfsu-corpus.org/channels/corpus)
    - `TED_zh_train.txt` 中文训练语料
    - `TED_en_train.txt` 英文训练语料
    - 以上两个语料，各行内容对应，如果出现中文或英文对应行去空格之后为空，则建议跳过该 pair
    - `TED_zh_test.txt` 中文测试语料
    - `TED_en_test.txt` 中文测试语料
    - 以上两个语料，各行内容对应，如果出现中文或英文对应行去空格之后为空，则建议跳过该 pair

测试方法：
- 在测试数据上评估训练好的模型的 cost (`sequence_loss`)。
