
# coding: utf-8

# In[56]:

import jieba
from jieba import posseg as pseg
from collections import defaultdict, Counter
import random
import numpy as np


# In[57]:

NEG = 'N'
POS = 'P'

train_files = {}
train_files[NEG] = 'neg_train.txt'
train_files[POS] = 'pos_train.txt'

test_files = {}
test_files[NEG] = 'neg_test.txt'
test_files[POS] = 'pos_test.txt'


# In[58]:

def load_file(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    segs = []        
    for line in lines:
        line = line.strip()
        words = pseg.cut(line)
        for (key, flag) in words:
            if flag == 'x':
                continue           
            segs.append(key)            
    return segs


# In[59]:

def load_train_dataset(input_data):
    segs = {}
    for k, v in input_data.items():
        segs[k] = load_file(v)
    return segs


# In[60]:

def calc_word_prob(segs):
    lm = defaultdict(Counter)
    for k,v in segs.items():
        for word in v:
            lm[k][word] += 1
    return lm


# In[61]:

def normalize_prob(lm_cnt):
   
    for key, cnt in lm_cnt.items():
        s = float(sum(cnt.values()))
        
        for word in cnt:
            cnt[word] /= s
    return lm_cnt    


# In[62]:

#计算每种情绪本身的概率 P(emotion)
def get_emotion_prob(lm):
    prob = {}    
    for k, v in lm.items():
        prob[k] = len(v)        
    
    s = float(sum(prob.values()))
    for k, v in prob.items():
        prob[k] /= s     
    return prob  


# In[87]:

def create_model(input_data):
    print 'Loading training data...'
    segs = load_train_dataset(input_data) #读入语料库，建立分词列表
    
    print 'Training data size by word:'
    for k, v in segs.items():
        print k, len(v)
        
    print 'Calculating word prob...'
    
    lm = calc_word_prob(segs) #统计词频, 计算 P(w|emotion)
    
    print 'Normalizing...'
    lm = normalize_prob(lm)   #归一化
    
    print 'Calculating emotion prob...'
    prob = get_emotion_prob(lm) #计算情绪独立概率 P(emotion)
    
    print 'Emotion prob:'
    for k,v in prob.items():
        print k, v
        
    print '-------- Model created---------'    
    for k,v in lm.items():        
        print k, len(v.items())
        #for word, prob in v.most_common()[:10]:
         #   print word, prob
    return lm, prob


# In[88]:

def load_file_by_lines(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    return lines


# In[89]:

def calc_line_prob(line_cnt, lm, e_p):
    s = [len(lm_dict[NEG].items()), len(lm_dict[POS].items())]
    p_neg = 1.0 * e_p[NEG] * s[0]
    p_pos = 1.0 * e_p[POS] * s[1]

    for word, v in line_cnt.items():        
        p_pos *= lm[POS][word] * s[1] * v
        p_neg *= lm[NEG][word] * s[0] * v
        
    if ((p_neg + p_pos) > 1.0e-6):        
        p_neg = p_neg / (p_neg + p_pos) 
        p_pos = 1.0 - p_neg
        
    return [p_neg, p_pos]


# In[90]:

def classify_line(line, lm, e_p):
    segs = pseg.cut(line.strip())
    cnt = Counter()
    for key, flag in segs:
        if flag == 'x':
            continue
        cnt[key] += 1    
    
    return calc_line_prob(cnt, lm, e_p)


# In[91]:

def classify_lines(lines, lm, e_p):
    tags = []
    for line in lines:
        tags.append(classify_line(line, lm, e_p))
    return tags    


# In[92]:

def calc_confusion_matrix(score_neg, score_pos):
    neg_cnt = len(score_neg)
    pos_cnt = len(score_pos)
    total_cnt = neg_cnt + pos_cnt
    
    neg = np.argmax(score_neg, 1)
    pos = np.argmin(score_pos, 1)
    
    fn = sum(neg)
    fp = sum(pos)
    
    error = 1.0 * (fn + fp) / total_cnt
    accuracy = 1.0 - error
    tp = pos_cnt - fp
    tn = neg_cnt - fn
    tp_rate = 1.0 - 1.0 * fp / pos_cnt
    fp_rate = 1.0 * fn / neg_cnt
    specity = 1.0 - fp_rate
    precision = 1.0 * tp /(pos_cnt + fn)
    prevalance = 1.0 * pos_cnt / total_cnt    
   
    return [total_cnt, neg_cnt, pos_cnt,
            tn, fn, tp, fp, accuracy, 
            tp_rate, fp_rate, specity,
            precision, prevalance]


# In[93]:

def load_test_dataset(files):
    segs = {}
    for k, v in files.items():
        segs[k] = load_file_by_lines(v)
    return segs


# In[94]:

def classify_test_data(lm, e_prob, test_data):
    probs = {}
    for k, v in test_data.items():
        probs[k] = classify_lines(test_data[k], lm, e_prob)
    return probs


# In[95]:

def calc_confusion_matrix(scores):
    score_neg = scores[NEG]
    score_pos = scores[POS]
    
    neg_cnt = len(score_neg)
    pos_cnt = len(score_pos)
    total_cnt = neg_cnt + pos_cnt
    
    neg = np.argmax(score_neg, 1)
    pos = np.argmin(score_pos, 1)
    
    fn = sum(neg)
    fp = sum(pos)    
    
    error = 1.0 * (fn + fp) / total_cnt
    accuracy = 1.0 - error
    tp = pos_cnt - fp
    tn = neg_cnt - fn
    tp_rate = 1.0 - 1.0 * fp / pos_cnt
    fp_rate = 1.0 * fn / neg_cnt
    specity = 1.0 - fp_rate
    precision = 1.0 * tp /(pos_cnt + fn)
    prevalance = 1.0 * pos_cnt / total_cnt
    
    matrix = {}
    
    matrix['Total'] = total_cnt
    matrix['Acutal N'] = neg_cnt
    matrix['Acutal P'] = pos_cnt    
    matrix['TP'] = tp
    matrix['TN'] = tn
    matrix['FP'] = fp
    matrix['FN'] = fn 
    matrix['Accuracy'] = accuracy
    matrix['TP rate'] = tp_rate
    matrix['FP rate'] = fp_rate
    matrix['Specity'] = specity
    matrix['Precision'] = precision
    matrix['Prevalance'] = prevalance
      
    return matrix


# In[100]:

def test_model(lm, e_prob, files):
    print 'Loading test dataset...'
    test_data = load_test_dataset(files)
    
    print 'Total test data:'
    for k,v in test_data.items():
        print k, len(v)
    
    print 'Classifing test data...'
    scores = classify_test_data(lm, e_prob, test_data)
    
    print 'Calcutating confusion matrix...'
    matrix = calc_confusion_matrix(scores)
    
    print '------- Test result --------'
    for k ,v in matrix.items():
        print k, v
    
    return matrix


# In[101]:

def print_confusion_matrix(matrix):
    print '------ Confusion matrix --------'
    tags = ['Total', 'Accuracy', 'Precision', 
            'TP', 'FP', 'TN', 'FN',
            'TP rate', 'FP rate',
           'Specity', 'Prevalance']
    for tag in tags:
        print tag, matrix[tag]


# In[98]:

lm_dict, e_prob = create_model(train_files)


# In[102]:

test_result = test_model(lm_dict, e_prob, test_files)


# In[103]:

print_confusion_matrix(test_result)


# In[ ]:



