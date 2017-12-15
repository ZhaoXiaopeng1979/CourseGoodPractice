
## 朴素贝叶斯法实现文本情感分类


```python
from collections import Counter
from math import log
import jieba

# (neg, pos) 的分类标记取为 (0, 1)，与各列表索引对应
train_files = ['_corpus/neg_train.txt', '_corpus/pos_train.txt']
test_files = ['_corpus/neg_test.txt', '_corpus/pos_test.txt']

def read_lines(file):
    with open(file, encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]
```

### 1. 依据训练集建立模型

#### 1.1 计算先验概率


```python
nums = [len(read_lines(train_files[c])) for c in (0, 1)]
prior = [nums[c]/sum(nums) for c in (0, 1)]  # 先验概率 neg, pos
prior
```




    [0.5288782233791589, 0.47112177662084115]



#### 1.2 计算条件概率


```python
def get_count_and_vocab(files=train_files):
    """读取训练数据，得到不同类别下的计数及词表"""
    count = [Counter(), Counter()]  # 计数：neg, pos
    vocab = set()  # 词表
    for c in (0, 1):
        for line in read_lines(files[c]):
            for word in jieba.cut(line):
                count[c][word] += 1
                vocab.add(word)
    return count, vocab

def to_log_prob(count, vocab):
    """将计数转换为条件概率，采用 Laplace add1 平滑"""
    log_conditional = [Counter(), Counter()]  # neg, pos
    vsize = len(vocab)
    for c in (0, 1):
        total = sum(count[c].values())
        for word in vocab:  # 这里必须是 vocab 而不是 count[c].keys()
            log_conditional[c][word] = log(count[c][word] + 1) - log(total + vsize)
    return log_conditional

count, vocab = get_count_and_vocab()
log_conditional = to_log_prob(count, vocab)
```

    Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    Loading model cost 0.745 seconds.
    Prefix dict has been built succesfully.


### 2. 对测试集数据进行分类预测



```python
def cal_joint_prob(docu, c):
    """计算文本 docu 与分类 c 的联合概率(取对数)"""
    log_joint_prob = log(prior[c])
    words = jieba.cut(docu)
    for word in words:
        if word in vocab:  # 参考 slp ch6.2, 仅考虑（训练集）词表内的词
            log_joint_prob += log_conditional[c][word]
    return log_joint_prob

def classify(docu):
    """对文本 docu 进行分类"""
    prob = [cal_joint_prob(docu, c) for c in (0, 1)]
    return 1 if prob[1] > prob[0] else 0

results = [[], []]  # 分类结果 neg, pos
for c in (0, 1):
    for line in read_lines(test_files[c]):
        results[c].append(classify(line))
```

### 3. 评估测试分类结果


```python
total = len(results[0]) + len(results[1])
neg_test_counter, pos_test_counter = Counter(results[0]), Counter(results[1])
true_pos = pos_test_counter[1]
false_pos = neg_test_counter[1]
true_neg = neg_test_counter[0]
false_neg = pos_test_counter[0]

# 混淆矩阵
confusion_matrix = ('\n'.join(['{:^15}' * 3] * 3)).format(
    'n='+str(total), 'predicted:neg', 'predicted:pos',
    'actual:neg', true_neg, false_pos,
    'actual:pos', false_neg, true_pos)
print(confusion_matrix)

# 评价指标：precision, recall, f_measure
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f_measure = 2 * precision * recall / (precision + recall)
print(('\nprecision = {:.3f}, recall = {:.3f}, f_measure = {:.3f}').format(
    precision, recall, f_measure))
```

        n=10538     predicted:neg  predicted:pos 
      actual:neg        4851            722      
      actual:pos        1095           3870      
    
    precision = 0.843, recall = 0.779, f_measure = 0.810

