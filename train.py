# coding: utf-8
import argparse
import pickle
import time
import math
import json
import collections as cl
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import model_rnn

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--unit', type=int, default=128, help='the number of lstm unit')
parser.add_argument('--layer', type=int, default=2, help='the number of layer')
parser.add_argument('--embed', type=int, default=32, help='embedding dimension')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.02, help='initial learning rate')
parser.add_argument('--bptt', type=int, default=32, help='sequence length')
parser.add_argument('--save', type=str,  default='model.pth', help='path to save the final model')
parser.add_argument('--cuda', type=bool, default=False, help='use CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Load data
vocab = pickle.load(open('data/vocab.pickle','r'))
train_data = pickle.load(open('data/train_data.pickle', 'r'))
val_data = pickle.load(open('data/val_data.pickle', 'r'))
train_data = torch.LongTensor(train_data)
val_data = torch.LongTensor(val_data)

# Build the model
# model = model_rnn.LSTM(len(vocab), train_data[0].shape[0]-1, 1)
model = model_rnn.LSTM(args.embed, len(vocab)-1, args.unit, args.layer)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), args.lr)

# Training code
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(args.cuda)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        output, hidden = model(data, hidden, cuda=args.cuda)
        total_loss += criterion(output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    start_time = time.time()
    hidden = model.init_hidden(args.cuda)
    # for i in tqdm(range(train_data.size(0))):
    for batch, i in enumerate(tqdm(range(0, train_data.size(0) - 1, args.bptt))):
        data, targets = get_batch(train_data, i)
        if len(data) != args.bptt:
            break
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()

        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden = model(data, hidden, cuda=args.cuda)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
setting = str(args.embed) + '_' +  str(args.unit) + '_' + str(args.layer) + '_minibatch' + str(args.bptt)
log = {"unit": args.unit, "layer": args.layer, "embed": args.embed, "loss": []}
loss_log = []
try:
    for epoch in range(1, args.epochs+1):
        if args.cuda:
            model =  model.cuda()
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        loss_log.append(val_loss)
        log["loss"] = loss_log
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:2.5f}s | valid loss {:2.5f} | '
            .format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.cpu(), open(setting + '.pth', 'wb'))
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
        
        # 10epochごとに保存
        if epoch%10 == 0:
            torch.save(model.cpu(), open(setting + "epoch" + str(epoch) + ".pth", "wb"))

        # lossを保存
        json.dump(log, open(setting + '.json', 'wb'))
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)

# Run on test data.
# test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)
