# coding: utf-8
import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import model_rnn

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.2, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
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
data = pickle.load(open('data/train_data.pickle', 'r'))
train_data = np.empty((0, data[0].size), dtype="float32")
val_data = np.empty((0, data[0].size), dtype="float32")
count = 0
for i in range(data.shape[0]):
  if count%10 == 0:
    val_data = np.append(val_data, np.array([data[i]]), axis=0)
  else:
    train_data = np.append(train_data, np.array([data[i]]), axis=0)
  count = count+1

train_data = torch.from_numpy(train_data).float()
val_data = torch.from_numpy(val_data).float()

# Build the model
model = model_rnn.LSTM(len(vocab), train_data[0].shape[0]-1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), args.lr)
if args.cuda:
   print("cuda!")
   model =  model.cuda()

# Training code
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(args.cuda)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data = Variable(train_data[i][:-1:])
        targets = Variable(train_data[i][1:])
	if args.cuda:
          data = data.cuda()
          targets = targets.cuda()
        output, hidden = model(data, hidden, cuda=args.cuda)
        total_loss += len(data) * criterion(output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    start_time = time.time()
    hidden = model.init_hidden(args.cuda)
    for i in range(train_data.size(0)):
        data = Variable(train_data[i][:-1])
        targets = Variable(train_data[i][1:])
        if args.cuda:
          data = data.cuda()
          targets = targets.cuda()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        # model.zero_grad()
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
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            .format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
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
