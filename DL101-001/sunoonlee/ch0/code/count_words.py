#!usr/bin/env python3
# coding: utf-8
"""分词并统计最高频的10个词"""

from collections import Counter
import jieba

def is_zh(word):
    """判断是否为汉字"""
    return '\u4e00' <= word[0] <= '\u9fff'

with open('happiness.txt') as f:
    text = f.read()

words = jieba.cut(text)
ct = Counter()
for w in words:
    if is_zh(w): 
        ct[w] += 1

print(ct.most_common(10))
