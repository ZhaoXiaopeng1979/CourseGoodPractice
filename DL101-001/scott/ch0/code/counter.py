# -*- coding: utf-8 -*-
import re
import click
import jieba
import zhon.hanzi as zh

from collections import Counter


class DataConuter(object):
    def __init__(self, s, replace_value=None):
        self.data = ' '.join(jieba.cut(s))
        self.replace_value = replace_value

    def clean_text(self):
        # 借助 Zhon 包，以中文标点符号切割
        line_list = re.split(r'[{}]'.format(zh.punctuation), self.data)
        cleared_line_list = [s.strip() for s in line_list]
        return cleared_line_list

    def count_words(self):
        terms = ' '.join(self.clean_text()).split()
        count = Counter(terms)
        value_sum = sum(count.values())
        return count, value_sum

    def bigrams(self):
        bigrams = [b for l in self.clean_text() for b in zip(l.split()[:-1], l.split()[1:])]
        # 把「的」之类的单个词去除
        # cleared_bigrams = [(a, b) for x, (a, b) in enumerate(bigrams) if len(a) > 1 and len(b) > 1]
        count = Counter(bigrams)
        value_sum = sum(count.values())
        return count, value_sum


@click.command()
@click.argument('inputfile')
@click.option('--replace-value', '-r')
@click.option('--bigram', '-b', is_flag=True)
@click.option('--top', '-t', default=10)
def cli(top, bigram, inputfile, replace_value):
    jieba.setLogLevel(20)
    with open(inputfile) as file:
        data = file.read()
    sample = DataConuter(data)
    if replace_value:
        if replace_value == 'de':
            replace_value = '的'
        count, value_sum = sample.count_words()
        probability = count[replace_value] / value_sum
        print('文章总词数为 {}，「{}」占比为 {:.2f}\n'.format(
            value_sum, replace_value, probability))
        print(data.replace(replace_value, ' **' + replace_value + '** '))

    else:
        if bigram:
            count, value_sum = sample.bigrams()
            top_counts = count.most_common(top)
            print('文章 Bigram 总数为 {}，出现频率最高的 {} 个二元词组是:'.format(
                value_sum, top))
            for (a, b), value in top_counts:
                print(a + ' ' + b, value)
        else:
            count, value_sum = sample.count_words()
            top_counts = count.most_common(top)
            print('文章总词数为 {}，出现频率最高的 {} 个词是:'.format(
                value_sum, top))
            for key, value in top_counts:
                print(key, value)


cli()
