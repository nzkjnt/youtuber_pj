# coding:utf-8
import numpy as np
import MeCab

import pickle
import codecs
import re

# テキストファイルを行ごとのリストとして読み込み
input_pickle = '../getMovie/youtuber.pickle'
# input_pickle = '../getMovie/test.pickle'
with open(input_pickle, "r") as f:
    titles = pickle.load(f)

titles = [ x.encode("utf-8") for x in titles]

# 行ごとに形態素に分解
m = MeCab.Tagger ("-O wakati")
words_by_titles = []
for title in titles:
    words = m.parse(title).split()
    words = ['<BOS>'] + words + ['<EOS>']
    words_by_titles.append(words)

# vocabを作る
vocab = {}
vocab["<BOS>"] = 0
vocab["<EOS>"] = 1
for i,words_by_title in enumerate(words_by_titles):
    for word in words_by_title:
        if word not in vocab:
            vocab[word] = len(vocab)

# 行ごとにvocabで表現
max_length = max([ len(x) for x in words_by_titles ])
dataset = np.empty((0, max_length), dtype="float32")
for i, words_by_title in enumerate(words_by_titles):
    datasetline = np.full(max_length, -1., dtype="float32")
    for j, word in enumerate(words_by_title):
        datasetline[j] = vocab[word]
    dataset = np.append(dataset, np.array([datasetline]), axis=0)

import os
if not os.path.exists("data"):
    os.mkdir("data")

print('line num:', len(dataset))
print('line_max_length:', max_length)
print('vocab size:', len(vocab))
pickle.dump(vocab, open('data/vocab.pickle', 'wb'))
pickle.dump(dataset, open('data/train_data.pickle', 'wb'))
