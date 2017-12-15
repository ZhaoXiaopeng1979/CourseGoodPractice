# coding: utf-8

# In[3]:

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq, embedding_rnn_seq2seq, sequence_loss, embedding_attention_seq2seq
from tensorflow.python.ops import variable_scope

import os
import time
from collections import Counter
import math
import jieba
from jieba import posseg as pseg
import numpy as np
import datetime
import pickle


DICT_FILE = './model/chat/dict_2.pkl'
MODEL_FILE = './model/chat/model_chat'
VOCAB_SIZE = 5000

def load_dict(file_name):
    wdict = {}
    wcnt = Counter()
    with open(file_name, 'rb') as f:
        (wdict, wcnt) = pickle.load(f)

    rdict = dict(zip(wdict.values(), wdict.keys()))
    return wcnt, wdict, rdict


#word_cnt, train_dict, train_reverse_dict = load_dict(DICT_FILE)

def pad_sentence(data, length, pad_index, start_index, end_index, is_encode=True):
    result_ = []
    data_len = len(data)
    if (data_len >= length):
        result_ = data[:length] #长句做截断处理
        if not is_encode:
            result_[length-1] = end_index
    else:
        pad_len = length - data_len
        padding = [pad_index] * pad_len
        if is_encode:
            result_ = padding + data
        else:
            result_ = [start_index] + data + [end_index] + padding[:-2]

    return result_

def get_batch_data2(offset, size, input_data, input_len, output_len):
    total_len = len(input_data)
    if (offset) > total_len:
        offset = 0
    if (offset+size>total_len):
        size = total_len - offset


    input_ = []
    output_ = []

    index = offset
    while(len(input_) < size):
        if (index >= total_len-1):
            index = 0
        if (len(input_data[index])>1):
            encode_data = pad_sentence(input_data[index], input_len, 0, START_ID, END_ID)
            decode_data = pad_sentence(input_data[index+1], output_len, 0, START_ID, END_ID, False)
            input_.append(encode_data)
            output_.append(decode_data)
        index += 1

    return input_, output_

def left_shift(decoder_inputs, pad_idx):
    # for generating targets
    return [list(input_[1:]) + [pad_idx] for input_ in decoder_inputs]


# In[18]:

def generate_feed_dict(batch_encoder_inputs, batch_decoder_inputs, pad_index,
        en_placeholders, de_placeholders, target_placeholders):
    encoder_inputs_ = list(zip(*batch_encoder_inputs))

    target_inputs_ = list(zip(*left_shift(batch_decoder_inputs, pad_index)))
    decoder_inputs_ = list(zip(*batch_decoder_inputs))

    feed_dict = dict()
    # Prepare input data
    for (i, placeholder) in enumerate(en_placeholders):
        # 这里用 placeholder 或者 placeholder.name 都可以
        feed_dict[placeholder.name] = np.asarray(encoder_inputs_[i], dtype=int)
        for i in range(len(de_placeholders)):
            feed_dict[de_placeholders[i].name] = np.asarray(decoder_inputs_[i], dtype=int)
            feed_dict[de_placeholders[i].name] = np.asarray(target_inputs_[i], dtype=int)
            # 这里使用 weights 把 <PAD> 的损失屏蔽了
            feed_dict[target_placeholders[i].name] = np.asarray([float(idx != pad_index) for idx in target_inputs_[i]],
                                                              dtype=float)
    return feed_dict


def index_to_words(data, dictionary, PAD_ID, END_ID):
    text = ''
    for w in data:
        if (w==END_ID):
            break
        if (w!=PAD_ID):
            text += dictionary[w]
    return text


def generate_response(session, test_sentence, wdict, rdict, encoder_len, decoder_len,
        PAD_ID, UNK_ID, START_ID, END_ID,
        cell, em_dim, vocab_size, en_placeholders, de_placeholders, target_placeholders):
    data = build_test_dataset(test_sentence, wdict, encoder_len, UNK_ID, START_ID, END_ID)
    output = decode_text(session, data, PAD_ID, decoder_len, cell, em_dim,
            vocab_size, en_placeholders, de_placeholders, target_placeholders)
    response = index_to_words(output, rdict, PAD_ID, END_ID)
    return response

def build_test_dataset(test_text, wdict, encoder_len, UNK_ID, START_ID, END_ID):
    words = test_text
    ids = words2id(words, wdict, UNK_ID)
    if (len(ids)>encoder_len):
        ids = ids[:encoder_len]
    return pad_sentence(ids, encoder_len, 0, START_ID, END_ID)


def words2id(words, wdict, UNK_ID):
    id_ = []
    for w in words:
        if w in wdict.keys():
            id_.append(wdict[w])
        else:
            id_.append(UNK_ID)
    return id_


# In[27]:

def decode_text(session, encode_input, pad_index, decoder_len, cell, em_dim,
        vocab_size, en_placeholders, de_placeholders, target_placeholders):
    # Decoding
    with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
        #outputs, states = embedding_rnn_seq2seq(
        outputs, states = embedding_attention_seq2seq(
             en_placeholders, de_placeholders, cell,
            vocab_size, vocab_size,
            em_dim, output_projection=None,
            feed_previous=True)

        decode_input = encode_input + [pad_index]*(decoder_len-len(encode_input))
        feed_dict_test = generate_feed_dict([encode_input], [decode_input], pad_index,
            en_placeholders, de_placeholders, target_placeholders)

        result = []
        for o in outputs:
            # 注意这里也需要提供 feed_dict
            m = np.argmax(o.eval(feed_dict_test, session=session), axis=1)
            result.append(m[0])

        return result



def chat(input_text):
    word_cnt, train_dict, train_reverse_dict = load_dict(DICT_FILE)

    LINE_BREAK = u'<Break>'
    WORD_DELIMITER = u'/'
    UNK_WORD = u'<UNK>'
    PADDING_WORD = u'<PAD>'
    START_WORD = u'<GO>'
    END_WORD = u'<EOS>'


    START_ID = train_dict[START_WORD]
    END_ID = train_dict[END_WORD]
    PAD_ID = train_dict[PADDING_WORD]
    UNK_ID = train_dict[UNK_WORD]

    #Attenion
    tf.reset_default_graph()

    RNN_CELL_TYPE = 'LSTMCell_Attention'
    learning_rate = 1.0

    encoder_length = 15
    decoder_length = 20
    embed_dim = 128

    cell = tf.contrib.rnn.LSTMCell(embed_dim)
    num_encoder_symbols = VOCAB_SIZE
    num_decoder_symbols = VOCAB_SIZE
    embedding_size = embed_dim

    encoder_len_placeholder = tf.placeholder(tf.int32)

    encoder_placeholders = [tf.placeholder(tf.int32, shape=[None],
                                           name="encoder_%d" % i) for i in range(encoder_length)]
    decoder_placeholders = [tf.placeholder(tf.int32, shape=[None],
                                           name="decoder_%d" % i) for i in range(decoder_length)]
    target_placeholders = [tf.placeholder(tf.int32, shape=[None],
                                           name="target_%d" % i) for i in range(decoder_length)]
    target_weights_placeholders = [tf.placeholder(tf.float32, shape=[None],
                                           name="decoder_weight_%d" % i) for i in range(decoder_length)]
    outputs, states = embedding_attention_seq2seq(
        encoder_placeholders, decoder_placeholders, cell,
        num_encoder_symbols, num_decoder_symbols,
        embedding_size, output_projection=None,
        feed_previous=False)

    loss = sequence_loss(outputs, target_placeholders, target_weights_placeholders)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)


    saver = tf.train.Saver()
    sess = tf.Session()

    #sess.run(tf.global_variables_initializer())
    saved_model = MODEL_FILE
    #print('Loading model from:', saved_model)

    #t0 = time.time()
    saver.restore(sess, saved_model)
    #t1 = time.time()
    #print(t1-t0)

    #input_text = u'你要去哪？'
    output_text = generate_response(sess, input_text, train_dict,
        train_reverse_dict, encoder_length, decoder_length,
        PAD_ID, UNK_ID, START_ID, END_ID,
        cell, embed_dim, VOCAB_SIZE,
        encoder_placeholders, decoder_placeholders, target_weights_placeholders)
    #print(output_text.encode("utf-8"))
    return output_text

#chat(u'你好')
