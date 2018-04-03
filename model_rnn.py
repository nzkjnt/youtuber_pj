# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
  def __init__(self, n_vocab, n_unit):
    super(LSTM, self).__init__()
    self.hiddensize = n_unit

    self.rnn = nn.LSTM(1, n_unit, 1)
    self.out = nn.Linear(n_unit, 1)

  def step(self, input, hidden=None):
    output, hidden = self.rnn(input.view(1, 1, -1), hidden)
    output = self.out(output.squeeze(1))
    return output, hidden

  def forward(self, inputs, hidden=None, force=True, steps=0, cuda=False):
    if force or steps == 0:
      steps = len(inputs)
    outputs = Variable(torch.zeros(steps))
    if cuda:
      outputs = outputs.cuda()
    for i in range(steps):
        if force or i == 0:
            input = inputs[i]
        else:
            input = output
        output, hidden = self.step(input, hidden)
        outputs[i] = output
    return outputs, hidden


  def init_hidden(self, cuda):
    hidden = (Variable(torch.zeros(1, 1, self.hiddensize)),
      Variable(torch.zeros(1, 1, self.hiddensize)))
    if cuda:
      return (hidden[0].cuda(), hidden[0].cuda())
    else:
      return hidden
