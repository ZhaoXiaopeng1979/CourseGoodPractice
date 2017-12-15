~ 主要用于存放本周视频中的示例代码 
- `conv_image.ipynb`: 卷积平滑图像的框架代码
- `lena512.png`: 上述代码的图像素材
- `conv_nlp.ipynb`: 卷积用于自然语言处理的示例代码

对 task3 脚本调用方式的要求

`python model.py train_shuffle.txt test_shuffle.txt`

`train_shuffle.txt` 及 `test_shuffle.txt` 是之前用过的情感分析语料，调整之后的格式为 `句子 \t label`，即句子和 label 用制表符分隔。`train_shuffle.txt` 为训练集，`test_shuffle.txt` 为测试集。
