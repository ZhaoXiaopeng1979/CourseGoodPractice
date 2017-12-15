
# coding: utf-8

# In[120]:

import jieba
from jieba import posseg as pseg
from collections import Counter, defaultdict
import random


# In[165]:

def load_file(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    segs = []
    for line in lines:
        line = line.strip().decode('utf-8')
        words = pseg.cut(line)
        words = [key for (key, flag) in words]
        for word in words:
            segs.append(word)            
    return segs


# In[166]:

def create_model(segs, n, padding):
    Delimiter = [u'。',u'！',u'？']
    lm = defaultdict(Counter)
    if n < 1:
        n = 1
    post_segs = []
    for i in range(n):
        post_segs.append(padding)
    for word in segs:
        post_segs.append(word)
        context = tuple(post_segs[-n-1:-1])
        lm[context][word] += 1
        if word in Delimiter:
            for j in range(n):
                post_segs.append(padding)  
                
    for key, cnt in lm.items():
        s = float(sum(cnt.values()))
        for word in cnt:
            cnt[word] /= s
    return lm
    


# In[167]:

def generate_head(n, pad):
    head = []
    for i in range(n):
       head.append(pad)
    return head


# In[168]:

def generate_word(lm_counter, context):
    r = random.random()
    s = 0.0
    for word, value in lm_counter[context].items():
        s += value
        if s > r:
            return word


# In[169]:

def generate_sentence(lm, start):    
    context = start
    sentence = []
    text = ''

    while (True):
        word = generate_word(lm, context)
        if word == None: 
            break
        else:
            text += word
            temp = list(context)[1:]
            temp.append(word)
            context = tuple(temp) 
            
    sentence.append(text.encode('utf-8'))
    return sentence


# In[170]:

def generate_sample_text(lm, n, sentence_count, padding):
    heads = generate_head(n, padding)
    start = tuple(heads)
    text = ''
    for i in range(sentence_count):
        sentences = generate_sentence(lm, start)
        sentences.append('\r\n')
        text += ''.join(sentences)
    return text


# In[171]:

def demo_model_from_file(file_name, max_n):
    segs = Load_File(file_name)
    padding = u'%'
    
    if max_n > 5:
        max_n = 5
    else:
        if max_n < 1:
            max_n = 1
            
    for n in range(1,max_n):
        print '-----------ngram = %d-----------' % (n)
        lm = create_model(segs, n, padding)
        text = generate_sample_text(lm, n, 10, padding)
        print text
        print '--------------------------------'


# In[173]:

demo_model_from_file('happiness.txt', 5)


# In[174]:

demo_model_from_file('LuXun.txt', 5)


# In[175]:

demo_model_from_file('HongLouMeng.txt', 5)


# In[176]:

demo_model_from_file('ZhangAiLing.txt', 5)




