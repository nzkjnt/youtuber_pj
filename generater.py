# coding:utf-8
import pickle
import torch
from torch.autograd import Variable


# 確率分布にしたがって，確率的に次の単語を出力
def next_word(model, hidden, word, max=False):
    output, hidden = model(Variable(torch.LongTensor([word])), hidden, False)
    output = output.data
    if max:
        v, idx = torch.max(output, 1)
    else:
        idx = torch.multinomial(output, 1)[0]

    return (idx[0], hidden)


# vocabのキーと値を入れ替えたdictを作成
vocab = pickle.load(open("data/vocab.pickle", "r"))
rvocab = {}
for key, value in vocab.items():
    rvocab[value] = key

# 学習済みモデルを展開
model = torch.load("cpu_64_128_2.pth")

# タイトル生成
for _ in range(10):
    hidden = model.init_hidden(False)

    result = [0]
    next = 0    # 最初の単語は<BOS>
    while next!=1:
        next, hidden = next_word(model, hidden, next)
        result.append(next)

    output = ""
    for idx in result:
        output += rvocab[idx]

    print(output)