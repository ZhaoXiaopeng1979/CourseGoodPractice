# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq, embedding_rnn_seq2seq, sequence_loss
from tensorflow.python.ops import variable_scope
import os
import time
from collections import Counter
import math
import jieba
from jieba import posseg as pseg
import numpy as np
import datetime

import chat_model

def response(input_text):
    result = chat_model.chat(input_text)
    return result
