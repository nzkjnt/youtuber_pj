# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, emb_dim, n_vocab, n_unit, n_layer, gpu, gpuid):
        super(LSTM, self).__init__()
        self.hiddensize = n_unit
        self.layersize = n_layer
        self.vocab = n_vocab
        self.gpu = gpu
        self.gpuid = gpuid

        self.embeddings = nn.Embedding(n_vocab, emb_dim)

        self.rnn = nn.LSTM(emb_dim, n_unit, n_layer)
        self.out = nn.Linear(n_unit, n_vocab)

    def step(self, input, hidden=None):
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        output = self.out(output.squeeze(1))
        output = F.log_softmax(output)
        return output, hidden

    def forward(self, inputs, hidden=None):
        outputs = Variable(torch.zeros(len(inputs), self.vocab))
        if self.gpu:
            outputs = outputs.cuda()

        embeds = self.embeddings(inputs)
        output, hidden = self.rnn(embeds.view(len(embeds), 1, -1), hidden)
        output = self.out(output.squeeze(1))
        output = F.log_softmax(output)
        return output, hidden

    def init_hidden(self):
        hidden = (Variable(torch.zeros(self.layersize, 1, self.hiddensize)),
            Variable(torch.zeros(self.layersize, 1, self.hiddensize)))
        if self.gpu:
            return (hidden[0].cuda(), hidden[0].cuda())
        else:
            return hidden
