#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing
import operator
import pickle
import string
from collections import Counter
import click
import jieba
import numpy as np
import pandas as pd
import zhon.hanzi as zh


def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, i, kwargs) for i, d in enumerate(
        np.array_split(df, workers))])
    pool.close()
    result = sorted(result, key=lambda x: x[0])
    return pd.concat([i[1] for i in result])


def other_multiprocessing(series, func, workers):
    chunk_size = int(series.shape[0] / workers)
    chunks = (series.ix[series.index[i:i + chunk_size]] for i in range(
        0, series.shape[0], chunk_size))

    pool = multiprocessing.Pool(processes=4)
    result = pool.map(func, chunks)
    return result


def save_dictionary(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def remove_stopwords(l, stopwords):
    return [s for s in l if s not in stopwords]


def remove_english_punctuation(l):
    return [s for s in l if s not in string.punctuation]


def remove_chinese_punctuation(l):
    return [s for s in l if s not in zh.punctuation]


def get_counter_sum(series):
    result_ = other_multiprocessing(series, np.sum, workers=4)
    counter_sum = np.sum(np.asarray(result_))
    return counter_sum


def get_common_words(count, freq):
    d = {i: j for i, j in count.items() if j > freq}
    sorted_dict = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_dict


def build_dict(word_counts):
    # Build word dictionary
    count = [['UNK', -1]]
    count.extend(word_counts)
    word_dict = {}
    for word, _ in count:
        word_dict[word] = len(word_dict)

    # Build reversed dictionary
    reversed_dict = {j: i for i, j in word_dict.items()}
    return word_dict, reversed_dict


def word_to_number(sentence, word_dict):
    # Word to number
    data = []
    for word in sentence:
        if word in word_dict:
            index = word_dict[word]
        else:
            index = 0
        data.append(index)
    return data


class DataClean(object):
    """Loads and clean the review dataset.

    Parameters
    ----------
    train_path: where the train dataset is, sentence and sentiment
        are delimited by '\t'.
    train_path: where the test dataset is, sentence and sentiment
        are delimited by '\t'.
    """
    def __init__(self, train_path, test_path, stopwords_path):
        self.train_path = train_path
        self.test_path = test_path
        self.stopwords_path = stopwords_path

    def read_data(self):
        """Read data into a DataFrame"""
        files = [self.train_path, self.test_path]
        dataset_class = ['train', 'test']
        self.dataframe = pd.DataFrame()
        list_ = []
        for i in range(2):
            df = pd.read_table(files[i], sep='\t', names='review sentiment'.split())
            df['dataset_class'] = dataset_class[i]
            list_.append(df)
        data = pd.concat(list_, ignore_index=True)
        return data

    def read_stopwords(self):
        with open(self.stopwords_path, 'r') as f:
            stop = f.read()
        stopwords = stop.split('\n')
        return stopwords

    def clean_data(self):
        """Clean the sentences.
        Parameters
        ----------
        self: object

        Returns
        -------
        df:
            columns: `cleared_words, sentiment, dataset_class,
                counter, word_counts, word_to_number`.
        """
        jieba.setLogLevel(20)
        jieba.enable_parallel(4)
        df = self.read_data()
        stopwords = self.read_stopwords()
        df['cut_words'] = df['review'].map(jieba.lcut)
        df['cleared_words'] = apply_by_multiprocessing(
            df['cut_words'], remove_english_punctuation, workers=4)
        df['cleared_words'] = apply_by_multiprocessing(
            df['cleared_words'], remove_chinese_punctuation, workers=4)
        df['cleared_words'] = apply_by_multiprocessing(
            df['cleared_words'], remove_stopwords, stopwords=stopwords, workers=4)
        df['counter'] = apply_by_multiprocessing(
            df['cleared_words'], Counter, workers=4)
        df['word_counts'] = apply_by_multiprocessing(
            df['cleared_words'], len, workers=4)
        columns = 'dataset_class sentiment cleared_words counter word_counts'
        df = df.loc[:, columns.split()]
        return df


def build_dataset(trian_path, test_path, stopwords_path, n, max_words):
    """Build the dataset for training and testing.
    Parameters
    ----------
    n: find the common words that count is over n.
    max_words: the sentences length.
    """
    # Get DataFrame of cleared words
    data = DataClean(trian_path, test_path, stopwords_path)
    df = data.clean_data()
    # Get the sum of df.counter
    couter_sum = get_counter_sum(df.counter)
    # Find the n most common words
    word_counts = get_common_words(couter_sum, n)
    # Build two dictionary
    word_dict, reversed_dict = build_dict(word_counts)
    # Save dictionary
    save_dictionary(word_dict, 'data/word_dict.pkl')
    save_dictionary(reversed_dict, 'data/reversed_dict.pkl')
    # word to number
    df['word_to_number'] = apply_by_multiprocessing(
        df.cleared_words, word_to_number, word_dict=word_dict, workers=4)

    # Build dataset
    train_df = df[df.dataset_class == 'train']
    test_df = df[df.dataset_class == 'test']
    text_data_train = train_df.word_to_number.values
    text_data_test = test_df.word_to_number.values
    # Pad/crop sentences to specific length
    text_data_train = np.array(
        [x[:max_words] for x in [y + [0] * max_words for y in text_data_train]])
    text_data_test = np.array(
        [x[:max_words] for x in [y + [0] * max_words for y in text_data_test]])
    target_train = train_df[['sentiment']].values
    target_test = test_df[['sentiment']].values

    # Save dataset
    train_data = np.concatenate((text_data_train, target_train), axis=1)
    test_data = np.concatenate((text_data_test, target_test), axis=1)

    np.savetxt('data/train_data.txt', train_data, fmt='%d')
    np.savetxt('data/test_data.txt', test_data, fmt='%d')
    print("Done!")
