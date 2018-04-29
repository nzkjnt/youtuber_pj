# coding:utf-8
import numpy as np
import MeCab
import os
import pickle

# テキストファイルを行ごとのリストとして読み込み
input_pickle = '../getMovie/youtuber.pickle'
# input_pickle = '../getMovie/test.pickle'
titles = pickle.load(open(input_pickle, "r"))
titles = [ x.encode("utf-8") for x in titles]
print("title num:", len(titles))

# 行ごとに形態素に分解
m = MeCab.Tagger ("-O wakati")
words_by_titles = []
for title in titles:
    words = m.parse(title).split()
    words = ['<BOS>'] + words + ['<EOS>']
    words_by_titles.append(words)

# vocabを作り，タイトルをvocabの数列で表現
vocab = {}
vocab["<BOS>"] = 0
vocab["<EOS>"] = 1
dataset = []
testset = []
title_count = 0
for i, words_by_title in enumerate(words_by_titles):
    title_count = title_count+1
    for word in words_by_title:
        if word not in vocab:
            vocab[word] = len(vocab)
        if title_count%10 == 0:
            testset.append(vocab[word])
        else:
            dataset.append(vocab[word])

if not os.path.exists("data"):
    os.mkdir("data")

print('dataset len:', len(dataset))
print('testset len:', len(testset))
print('vocab size:', len(vocab))
pickle.dump(vocab, open('data/vocab.pickle', 'wb'))
pickle.dump(dataset, open('data/train_data.pickle', 'wb'))
pickle.dump(testset, open('data/val_data.pickle', 'wb'))
