# coding:utf-8
import math
import pickle
import numpy as np
import torch
from torch.autograd import Variable


# 確率分布にしたがって，確率的に次の単語を出力
def next_word(model, hidden, word, max=False):
    output, hidden = model(Variable(torch.LongTensor([word])), hidden, False)
    output = output.data.numpy()
    if max:
        idx = np.argmax(output)
    else:
        output = torch.FloatTensor([math.exp(x) for x in output[0]])
        idx = torch.multinomial(output, 1)[0]

    return (idx, hidden)


# vocabのキーと値を入れ替えたdictを作成
vocab = pickle.load(open("data/vocab.pickle", "r"))
rvocab = {}
for key, value in vocab.items():
    rvocab[value] = key

# 学習済みモデルを展開
model = torch.load("model/64_128_1_0.2_minibatch128_epoch90.pth")

# タイトル生成
for _ in range(10):
    hidden = model.init_hidden(False)

    result = []
    next = 0    # 最初の単語は<BOS>
    MAX_PROB = False # 確率最大の単語を取り出す
    while next!=1:
        next, hidden = next_word(model, hidden, next, MAX_PROB)
        # MAX_PROB = not MAX_PROB
        # print(rvocab[next])
        result.append(next)

    output = ""
    for idx in result[:-1]:
        output += rvocab[idx]

    print(output)