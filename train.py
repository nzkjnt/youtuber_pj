# coding: utf-8
import os
import argparse
import pickle
import time
import math
import json
from tqdm import tqdm

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
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--g', type=int, default=0, help='gpu id')
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
model = model_rnn.LSTM(args.embed, len(vocab)-1, args.unit, args.layer, args.cuda, args.g)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), args.lr)

if args.cuda:
    model =  model.cuda(args.g)
    train_data = train_data.cuda(args.g)
    val_data = val_data.cuda(args.g)

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

def evaluate(input, target, hidden):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = repackage_hidden(hidden)
    output, hidden = model(data, hidden)
    loss = criterion(output, targets)
    
    return loss.data[0], hidden

def train(input, target, hidden):
    # Turn on training mode which enables dropout.
    model.train()
    hidden = repackage_hidden(hidden)
    optimizer.zero_grad()
    output, hidden = model(input, hidden)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    for p in model.parameters():
        p.data.add_(-lr, p.grad.data)

    return loss.data[0], hidden


# 保存用ディレクトリ作成
if not os.path.exists('./model'):
    os.mkdir('./model')
if not os.path.exists('./loss'):
    os.mkdir('./loss')

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
setting = str(args.embed) + '_' +  str(args.unit) + '_' + str(args.layer) + '_' + str(args.lr) + '_minibatch' + str(args.bptt)
log = {"unit": args.unit, "layer": args.layer, "embed": args.embed, "testloss": [], "trainloss": []}
train_loss = []
test_loss = []
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        if args.cuda:
            model =  model.cuda(args.g)

        # train
        epoch_loss = 0
        hidden = model.init_hidden()
        for batch, i in enumerate(tqdm(range(0, train_data.size(0) - 1, args.bptt))):
            data, targets = get_batch(train_data, i)
            if len(data) != args.bptt:
                break
            loss, hidden = train(data, targets, hidden)
            epoch_loss += loss
        epoch_loss /= len(train_data)

        # evaluation
        val_loss = 0
        hidden = model.init_hidden()
        for i in range(0, val_data.size(0) - 1, args.bptt):
            data, targets = get_batch(val_data, i, evaluation=True)
            loss, hidden = evaluate(data, targets, hidden)
            val_loss += loss
        val_loss /= len(val_data)

        test_loss.append(val_loss)
        train_loss.append(epoch_loss)
        log["testloss"] = test_loss
        log["trainloss"] = train_loss

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:2.5f}s | valid loss {:2.5f} | '
            .format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.cpu(), open('./model/' + setting + '.pth', 'wb'))
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
        
        # 10epochごとに保存
        if epoch%10 == 0:
            torch.save(model.cpu(), open('./model/' + setting + "_epoch" + str(epoch) + ".pth", "wb"))

        # lossを保存
        json.dump(log, open('./loss/' + setting + '.json', 'wb'))
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
