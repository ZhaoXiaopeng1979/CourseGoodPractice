# -*- coding: utf-8 -*-
import random
import jieba
from collections import Counter


def jieba_cut(filename):
    """Return list with jieba.cut."""
    jieba.enable_parallel(4)
    with open(filename, 'r') as f:
        data = f.read()
        lst = [i for i in jieba.cut(data)]
    return lst


def find_ngrams(lst, n):
    return list(zip(*[lst[i:] for i in range(n)]))


def normalization(lst):
    cnt = Counter(lst)
    s = sum(cnt.values())
    for key, value in cnt.items():
        cnt[key] /= s
    return cnt


def generate_word(cnt):
    r = random.random()
    s_ = 0.0
    for word, prob in cnt.items():
        s_ += prob
        if s_ >= r:
            return word


def generate_text(cnt):
    s = ''
    for i in range(100):
        word = ''.join(generate_word(cnt))
        s += word
    print(s)
