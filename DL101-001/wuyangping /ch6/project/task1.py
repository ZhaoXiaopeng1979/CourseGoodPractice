
# coding: utf-8

# In[1]:

get_ipython().magic(u'load_ext watermark')
get_ipython().magic(u'watermark -p tensorflow,numpy -v -m')


# In[2]:

import tensorflow as tf
import numpy as np
import os
print(tf.__version__)


# In[3]:

get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
import jieba
from jieba import posseg as pseg
from collections import defaultdict, Counter
import random
import time


# In[4]:

NEG = 'N'
POS = 'P'

train_files = {}
train_files[NEG] = 'neg_train.txt'
train_files[POS] = 'pos_train.txt'

test_files = {}
test_files[NEG] = 'neg_test.txt'
test_files[POS] = 'pos_test.txt'


# In[5]:

#读取文件，分词
def load_file(file_name, line_num=0):
    f = open(file_name, 'r')    
    lines = f.readlines()
    cnt = len(lines)
    if line_num >0:
        cnt = line_num
    f.close()
    segs = []
    seg_lines = []
    for line in lines[:cnt]:
        line = line.strip()
        words = pseg.cut(line)
        seg_per_line = []
        for (key, flag) in words:
            if flag == 'x':
                continue           
            segs.append(key)
            seg_per_line.append(key)
        seg_lines.append(seg_per_line)
    return segs, seg_lines


# In[6]:

def load_train_dataset(input_data, line_num=0):
    segs = {}
    seg_lines = {}
    for k, v in input_data.items():
         segs[k], seg_lines[k] = load_file(v, line_num)
    return segs, seg_lines


# In[7]:

t0 = time.time()

segs_dict, seg_lines_dict = load_train_dataset(train_files)

t1 = time.time()
print(t1-t0)


# In[8]:

print(len(segs_dict[NEG]))
print(len(segs_dict[POS]))
print(len(seg_lines_dict[NEG]))
print(len(seg_lines_dict[POS]))


# In[9]:

UNKNOWN_WORD = u'UNK'


# In[10]:

#建立vocabulary dict
def build_word_dict(input_segs):
    all_segs = []
    temp = []
    for (k, v) in input_segs.items():
        all_segs.extend(v)
    word_cnt = Counter(all_segs)
    word_dict = {}
    word_dict[UNKNOWN_WORD] = 0
    index_dict = {}
    index_dict[0] = UNKNOWN_WORD
    i = 1
    for (k, v) in word_cnt.most_common()[:4999]:
        word_dict[k] = i
        index_dict[i] = k
        i += 1
    return word_dict, index_dict


# In[11]:

all_word_dict, all_index_dict = build_word_dict(segs_dict)


# In[13]:

#把语句转换为词索引
def build_line_data(lines, word_dict, isTestdata=False):
    lines_index = {}
    labels_index = {}
    i = 0
    max_len = 0
    for (k,v) in lines.items():
        label = 0
        if (k==POS):
            label = 1
        for line in v:
            seg_index = []
            labels_index[i] = label            
            for word in line:
                if isTestdata:                    
                    if word in word_dict.keys():
                        seg_index.append(word_dict[word])
                    else:
                        seg_index.append(0)
                else:
                    seg_index.append(word_dict[word])
            lines_index[i] = seg_index
            if (max_len < len(seg_index)):
                max_len = len(seg_index)                
            i+=1
    return max_len, lines_index, labels_index


# In[14]:

t0 = time.time()
max_sentence_len, train_sentences, train_labels = build_line_data(seg_lines_dict, all_word_dict, True)
t1 = time.time()
print(t1-t0)


# In[15]:

print(max_sentence_len)
print(len(train_sentences))
print(len(train_labels))

i=0
#句子长度设为60，训练语料中80%句子长度小于60
SENTENCE_LEN=60

for (k,s) in train_sentences.items():
    
    if len(s)<=SENTENCE_LEN:
        i+=1

total_sentence = len(train_sentences)
print i
outsider = total_sentence - i
print outsider
print 100.0 * outsider / total_sentence


# In[16]:

#把每行语料变成固定长度，短句后面补未命中词，长句直接截断
def build_input_train_data(sentences, max_len):
    input_ = {}
    for (k,v) in sentences.items():
        input_[k] = v[:max_len]
        if (len(v) < max_len):
            padding = [0] *(max_len-len(v))
            input_[k].extend(padding)
    return input_


# In[17]:

input_train_data = build_input_train_data(train_sentences, SENTENCE_LEN)


# In[20]:

#读入测试语料
t0 = time.time()
test_segs, test_lines = load_train_dataset(test_files,1000)
t1 = time.time()
print(t1-t0)


# In[22]:

t0 = time.time()
s_len, test_sentences, test_labels = build_line_data(test_lines, all_word_dict,isTestdata=True)
t1 = time.time()
print(t1-t0)


# In[24]:

print(s_len)
print(len(test_sentences))
print(len(test_labels))


# In[25]:

input_test_data = build_input_train_data(test_sentences, SENTENCE_LEN)


# In[26]:

vocab_size = len(all_word_dict)
word_embed_size = 64

print(vocab_size)


# In[27]:

#将label变成one-hot matrix格式
def get_label_matrix(input_label, num_l):
    out_ = []
    for label in input_label:
        line = [0] * num_l
        line[label] = 1
        out_.append(line)    
    return out_


# In[28]:

def get_all_test_data(data, labels):
    output_ = []
    labels_ = []
    for (k,v) in data.items():
        output_.append(v)
    for (k,v) in labels.items():
        labels_.append(v)
    return output_, get_label_matrix(labels_, 2)


# In[30]:

def shuffle_data(input_data, input_labels):
    output_data = []
    for (index,v) in input_data.items():
        label = input_labels[index]
        output_data.append((index, label, v))
    np.random.shuffle(output_data)
    return output_data


# In[31]:

#对输入数据作shuffle处理
shuffled_train_data = shuffle_data(input_train_data, train_labels)
shuffled_test_data = shuffle_data(input_test_data, test_labels)

print(len(shuffled_train_data))
print(len(shuffled_test_data))


# In[41]:

#获取单批训练/测试数据
def get_batch_data(input_data, index, size, num_l):
    data_ = []
    labels_ = []
    indexs_ = []
    for (i, k,v) in input_data[index:index+size]:
        data_.append(v)
        labels_.append(k)
        indexs_.append(i)
    return indexs_, data_, get_label_matrix(labels_, num_l)


# In[42]:

def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[43]:

tf.reset_default_graph()


# In[44]:

tf.reset_default_graph()

graph = tf.Graph()
sentence_length = SENTENCE_LEN
# max_pool
with graph.as_default():
    filter_num = 64
    window_size = 3
    num_labels = 2
    num_fc_hidden = 10
    
    tf_input_data = tf.placeholder(tf.int32, shape=[None, sentence_length], name='input_data')    
    tf_labels = tf.placeholder(tf.int32, shape=[None, num_labels], name='labels')
    
    word_embeds = tf.Variable(tf.random_uniform([vocab_size, word_embed_size], -1.0, 1.0), name="Word_embed")
    input_embeds = tf.nn.embedding_lookup(word_embeds, tf_input_data)
   
    tf_embeds_expand = tf.expand_dims(input_embeds, -1)
    
    print(tf_input_data)
    print(tf_labels)
    print(input_embeds)
    print(tf_embeds_expand)

    filter_shape = [window_size, word_embed_size, 1, filter_num]
    # W 和 b 是卷积的参数
    W = tf.Variable(tf.random_uniform(filter_shape, -1.0, 1.0), name="W")
    # bias 和 filter_num 个数是一样的
    b = tf.Variable(tf.constant(0.0, shape=[filter_num]), name="b")
    # 步长为1，这里不做 Padding，因此句子太短的话可能要丢掉。可自行尝试加 padding（不加也不影响作业评分）
    conv = tf.nn.conv2d(
                    tf_embeds_expand,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
    # 卷积出来的结果加上 bias
    conv_hidden = tf.nn.tanh(tf.add(conv, b), name="tanh")
    
    print(conv)

    # 因为没有 padding，出来的结果个数是 sequence_length - window_size + 1，如果加了 padding 这里要对应更改。
    pool = tf.nn.max_pool(
                    conv_hidden,
                    ksize=[1, sentence_length - window_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
    
    print(pool)
    
    #增加一个全连接层
    fc = tf.layers.dense(pool, num_fc_hidden, activation=tf.nn.tanh)
    
    print(fc)
   
    raw_output = tf.layers.dense(fc, num_labels, name='output')
    print(raw_output)
    
    
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=raw_output, labels=tf_labels)
    
    cost_summary = tf.summary.scalar('cost', tf.reduce_mean(cost))
    embed_summary = tf.summary.histogram('embed',input_embeds)
    merged = tf.summary.merge_all()

    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cost)
    
    tf_prediction = tf.nn.softmax(raw_output)


# In[45]:

writer = tf.summary.FileWriter("/root/log")


# In[51]:

def train_model(num_epoch, batch_size_num):
    num_labels = 2
    total_train_batch = len(shuffled_train_data) / batch_size_num
    if len(input_train_data) % batch_size_num > 0:
        total_train_batch += 1
   
    total_test_batch = len(shuffled_test_data) / batch_size_num
    if len(shuffled_test_data) % batch_size_num > 0:
        total_test_batch += 1
  
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
       
        train_start = time.time()
        costs = []
        for epoch in range(num_epoch):
            start_index = 0

            train_acc = []
            for i in range(total_train_batch):            
                batch_index, batch_data, batch_labels = get_batch_data(shuffled_train_data, start_index, batch_size_num, num_labels)

                start_index += batch_size_num

                feed_dict = {tf_input_data : batch_data, tf_labels : batch_labels}
                _, c, predictions = session.run(
                  [train_step, cost, tf_prediction], feed_dict=feed_dict)

                acc = accuracy(np.reshape(predictions,[len(batch_labels),num_labels]), batch_labels)
                train_acc.append(np.mean(acc))
                
            costs.append(np.mean(c))

        train_end = time.time()
        duration = (train_end - train_start)
        
        print("Batch size=%d" % batch_size_num)
        print("Epoches=%d" % num_epoch)
        print("Training duration=%.2f" % duration)
        print("Training accuracy=%.2f" % train_acc[-1])
        print("Training cost=%.2f" % costs[-1])
        
        return duration, train_acc


# In[52]:

t0, acc0 =  train_model(1, 125)


# In[53]:

t0, acc0 =  train_model(1, 1)


# In[54]:

t0, acc0 =  train_model(10, 125)


# In[55]:

t0, acc0 =  train_model(10, 1)


# # Summary
# 
#              Epoches=1  Epoches=10
# Batch size = 1     29       312
# Batch size = 125    11       109
# 
# 其他数据不变的情况下，Batch size从1改成125，训练时间可以节约65%左右
