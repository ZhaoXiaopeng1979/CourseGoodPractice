
# coding: utf-8

# ## 运行环境说明：

# In[65]:

import sys
import platform

print("系统版本是：{}".format(' '.join(platform.linux_distribution())))
print("Python 版本是：{}".format(sys.version.split()[0]))


# # 2w 作业
#
# * 任务1：贝叶斯公式的运用
# 	利用贝叶斯公式，说明为什么 $P(y|w1, w2) ≠ P(y|w1)P(y|w2)$（即使做了独立假设）
#
# * 任务2：实现 Naive Bayes 方法
# 	请你用 Python 实现 Naive Bayes 方法，并在给定的数据集上验证数据。具体要求如下：
#     在「训练数据」上拟合一个 Naive Bayes 模型。在训练时模型不能「看见」任何测试数据的信息。
#     训练完成后，在测试数据上进行测试。评估标准为你的模型在测试数据上的混淆矩阵（Confusion Matrix）结果。
#     根据混淆矩阵的结果，分析一下你模型的表现。
# 	参考概念：混淆矩阵 Simple guide to confusion matrix terminology
#
# * 任务3：实现 Gradient Descent 算法
# 	通过梯度下降法，自己实现一种通用的给定数据找到 y = wx + b 中最优的 w 和 b 的程序，并用加噪音数据验证效果。
#

# ## 1. 贝叶斯公式的运用
#
# $P(y|w_1) = \frac{P(w_1|y)P(y)}{P(w_1)}$
#
# $P(y|w_2) = \frac{P(w_2|y)P(y)}{p(w_2)}$
#
# $P(y|w_1)P(y|w_2) = \frac{P(w_1|y)P(w_2|y)P(y)^2}{p(w_1)P(w_2)}$
#
# $P(y|w_1,w_2) = \frac{P(w_1,w_2|y)p(y)}{P(w_1,w_2)}$
#
# 从上面两个公式可以看出，哪怕做了独立假设，
#
# $P(y|w_1).P(y|w_2) 也是不等于 $P(y|w_1,w_2)

# ## 2. 实现 Naive Bayes 方法
#
# ### 贝叶斯公式和应用：
#
# 贝叶斯的核心是通过先验概率和逆条件概率从而求出条件概率：
#
# * 正面和负面分别占比，是其中之一的先验概率
# * 已知文本的情感，各个词所占的概率是逆条件概率
#
# ### 作业思路和解决步骤：
#
# 1. 分词，并剔除停止词(影响情感)
# 2. 统计各个词在正负面所占的概率，并计算他们的联合概率(考虑独立性)
# 3. 生成模型(还要细化。。。)
#
#
# ### Refrences:
#
# * [Bayesian Classification withInsect examples](http://www.cs.ucr.edu/~eamonn/CE/Bayesian%20Classification%20withInsect_examples.pdf) 这个 Slide 超赞

# ### 2.1 读取并分词

# In[1]:

import os
import string
import re
import pyprind
import multiprocessing
import jieba
import numpy as np
import pandas as pd

from collections import Counter


# 首先用 linux 命令简单查看下数据

# In[2]:



# 发现数据每一行都是针对不同商品且不一样的评论，所以一行其实就是一个情绪。

# 参考 [rasbt/python-machine-learning-book](https://github.com/rasbt/python-machine-learning-book) 把每行读取出来，且生成标签

# In[3]:

basepath = './data'

labels = {'pos_train': 1, 'neg_train':0, 'neg_test': '0?', 'pos_test': '1?'}
pbar = pyprind.ProgBar(36000)  # 迭代次数
df = pd.DataFrame()
for i in labels:
    path = os.path.join(basepath, i + '.txt')
    with open(path) as f:
        lines = (line.strip() for line in f.readlines())
    for line in lines:
        df = df.append([[line, labels[i]]], ignore_index=True)
        pbar.update()
df.columns = ['review', 'sentiment']


# In[4]:

df.head()


# #### 分词：
#
# 选择一种多进程的 Apply 方式来分词

# In[5]:

def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.apply(func, **kwargs)

def apply_by_multiprocessing(df,func,**kwargs):
    workers=kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, i, kwargs) for i,d in enumerate(np.array_split(df, workers))])
    pool.close()
    result=sorted(result,key=lambda x:x[0])
    return pd.concat([i[1] for i in result])


# In[6]:

df['cut_words'] = apply_by_multiprocessing(df.review, jieba.lcut, workers=4)


# In[7]:

df.tail()


# 停止词对于情感分析毫无帮助，所以剔除

# In[8]:

with open('data/stop_words_chinese.txt') as file:
    data = file.read()


# In[9]:

stop_words_chinese = data.split('\n')


# In[10]:

def remove_stop_words(l): return [s for s in l if s not in stop_words_chinese]


# In[11]:

df['cleared_words'] = apply_by_multiprocessing(df.cut_words, remove_stop_words, workers=4)


# 移除中文停止词之后，发现还有英文的标点符号

# In[12]:

def remove_english_punctuation(l): return [s for s in l if s not in string.punctuation ]


# In[13]:

df['cleared_words'] = apply_by_multiprocessing(df.cleared_words, remove_english_punctuation, workers=4)


# In[14]:

df.head()


# In[15]:

df['counter'] = apply_by_multiprocessing(df.cleared_words, Counter, workers=4)


# 数据长度，方便计算总长度

# In[16]:

df['words_count'] = df.cleared_words.map(len)


# In[17]:

df.head(2)


# In[18]:
df.to_pickle('data/cleared_data.pkl')
#df.to_csv('data/cleared_data.csv', index=False)
# df_train = df[(df.sentiment == 1 )| (df.sentiment ==0)]
# df_train_pos =  df[df.sentiment == 1]
# df_train_neg = df[df.sentiment == 0]
# df_test_pos = df[df.sentiment == '1?']
# df_test_neg = df[df.sentiment == '0?']

