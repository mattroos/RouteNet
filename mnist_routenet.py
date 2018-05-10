# mnist_routenet.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import time
import sys
import ConfigParser
import matplotlib.pyplot as plt

# from routenet import RouteNet as RN
import routenet as rn

plt.ion()

# IDEA: Can we pretrain without gating so batches can be used? And then
# do additional training with gates? Or, could keep gates, but use as
# soft gates.
# TODO: Measure how many gates are active on average, and report during training.
# TODO: Add activation loss. All activations or just gates?
# TODO: Hard sigmoid gate activation function, and probabilistic gating.
# TODO: Hierarchical routing?  Fractal/hierarchical connectivity patterns/modularity?
# TODO: Accompanying mechanisms to modulate learning?


# Read in path where raw and processed data are stored
configParser = ConfigParser.RawConfigParser()
configParser.readfp(open(r'config.txt'))
dirMnistData = configParser.get('Data Directories', 'dirMnistData')


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lambda-nll', type=float, default=1.0, metavar='N',
                    help='weighting on nll loss. weight on activation loss is 1-lambda_nll.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

lambda_nll = args.lambda_nll
lambda_act = 1 - args.lambda_nll

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dirMnistData, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dirMnistData, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


banks_per_layer = np.asarray([10,10])
bank_conn = rn.make_conn_matrix(banks_per_layer)
model = rn.RouteNet(n_input_neurons=784, \
                 idx_input_banks=np.arange(banks_per_layer[0]), \
                 bank_conn=bank_conn, \
                 idx_output_banks=np.arange( np.sum(banks_per_layer)-banks_per_layer[-1], np.sum(banks_per_layer) ), \
                 n_output_neurons=10, \
                 n_neurons_per_hidd_bank=5)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    loss_sum = 0.0
    prob_open_gate_sum = 0.0
    cnt = 0
    t_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # act_fc1, act_fc2, act_fc3, output = model(data)
        output, prob_open_gate = model(data)

        if output is not None:
            loss_nll = F.nll_loss(output, target)
            # loss_act = torch.mean(act_fc3) + torch.mean(act_fc2) + torch.mean(act_fc1)
            # loss_act = torch.mean(act_fc3) + torch.mean(act_fc2)
            # loss = lambda_nll*loss_nll + lambda_act*loss_act
            loss = loss_nll

            loss.backward()
            optimizer.step()
            # TODO?: Currently have to use batch size of one. Could we accumulate
            # gradients over a number of samples and then update weights, without
            # using the optimizer? Can't use usual pytorch approach because
            # the constructed graph can be difference for each sample, during
            # training.

            loss_sum = loss_sum + loss.data.cpu().numpy()[0]
            prob_open_gate_sum += prob_open_gate
            cnt += args.batch_size

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%), {:d} valid]\tLoss: {:.6f}\tProb open gate: {:.2f}\t{:.2f} seconds'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.data[0]))
                100. * batch_idx / len(train_loader), cnt, loss_sum/cnt, prob_open_gate_sum/cnt, time.time()-t_start))
            loss_sum = 0.0
            prob_open_gate_sum = 0.0
            cnt = 0
            t_start = time.time()

def test():
    model.eval()
    test_loss_nll = 0
    test_loss_act = 0
    correct = 0
    cnt = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        #act_fc1, act_fc2, act_fc3, output = model(data)
        output, _ = model(data)

        if output is not None:
            test_loss_nll += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            # test_loss_act = (torch.mean(act_fc3) + torch.mean(act_fc2) + torch.mean(act_fc1)).data[0]
            # test_loss_act = (torch.mean(act_fc3) + torch.mean(act_fc2)).data[0]
            test_loss = test_loss_nll

            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            cnt += args.test_batch_size

    # test_loss_nll /= len(test_loader.dataset)
    # test_loss_act /= len(test_loader.dataset)
    # test_loss = lambda_nll * test_loss_nll + lambda_act * test_loss_act
    test_loss_nll /= cnt
    print('\nTest bank: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss_nll, test_loss_act


## Run it
loss_nll = np.zeros(args.epochs)
loss_act = np.zeros(args.epochs)
t_start = time.time()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    loss_nll[epoch-1], loss_act[epoch-1] = test()

dur = time.time()-t_start
print('Time = %f, %f sec/epoch' % (dur, dur/args.epochs))

fn = 1

## Plot losses for test bank
plt.figure(fn)
fn = fn + 1
plt.clf()

plt.subplot(3,1,1)
plt.semilogy(loss_nll,'bo-')
plt.title('NLL loss')
plt.xlabel('Epoch')
plt.grid()

plt.subplot(3,1,2)
plt.semilogy(loss_act, 'o-')
plt.title('Activation loss')
plt.xlabel('Epoch')
plt.grid()

plt.subplot(3,1,3)
plt.semilogy(lambda_act * loss_act + lambda_nll * loss_nll, 'ro-')
plt.title('Total loss')
plt.xlabel('Epoch')
plt.grid()

## Plot the weights
weights = []
grads = []
for param in model.parameters():
    weights.append(param.data)
    grads.append(param.grad)
plt.figure(fn)
fn = fn + 1
plt.clf()
for i in range(0, len(weights)/2):
    plt.subplot(2,2,i+1)
    plt.imshow(weights[2*i], aspect='auto', interpolation='nearest')
    plt.colorbar()

## Plot activations for batch
model.eval()
test_loss_nll = 0
test_loss_act = 0
correct = 0
for data, target in test_loader:
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    act_fc1, act_fc2, act_fc3, output = model(data)

    mx = torch.max(act_fc3).data[0]
    mx = max(mx, torch.max(act_fc2).data[0])
    mx = max(mx, torch.max(act_fc1).data[0])

    plt.figure(fn)
    fn = fn + 1
    n_samps_display = 30
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(act_fc1.data.cpu().numpy()[:n_samps_display,:], aspect='auto', interpolation='nearest')
    plt.clim(0, mx)
    plt.subplot(2,2,2)
    plt.imshow(act_fc2.data.cpu().numpy()[:n_samps_display,:], aspect='auto', interpolation='nearest')
    plt.clim(0, mx)
    plt.subplot(2,2,3)
    plt.imshow(act_fc3.data.cpu().numpy()[:n_samps_display,:], aspect='auto', interpolation='nearest')
    plt.clim(0, mx)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(output.data.cpu().numpy()[:n_samps_display,:], aspect='auto', interpolation='nearest')
    # plt.clim(0, 1)

    plt.figure(fn)
    fn = fn + 1
    n_samps_display = 30
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(act_fc1.data.cpu().numpy()[:n_samps_display,:]>0, aspect='auto', interpolation='nearest')
    plt.subplot(2,2,2)
    plt.imshow(act_fc2.data.cpu().numpy()[:n_samps_display,:]>0, aspect='auto', interpolation='nearest')
    plt.subplot(2,2,3)
    plt.imshow(act_fc3.data.cpu().numpy()[:n_samps_display,:]>0, aspect='auto', interpolation='nearest')
    plt.subplot(2,2,4)
    plt.imshow(output.data.cpu().numpy()[:n_samps_display,:], aspect='auto', interpolation='nearest')

    ## Plot activations for some individual classes: E.g., 0, 1, 2 ,3
    ## Do we see trends in the activation patterns at higher layers?
    for i_class in range(3):
        ix = np.where(target.data.cpu().numpy()==i_class)[0]
        ix = ix[0:n_samps_display]
        plt.figure(fn)
        fn = fn + 1
        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(act_fc1.data.cpu().numpy()[ix,:]>0, aspect='auto', interpolation='nearest')
        plt.title('Class #%d' % (i_class))
        plt.subplot(2,2,2)
        plt.imshow(act_fc2.data.cpu().numpy()[ix,:]>0, aspect='auto', interpolation='nearest')
        plt.subplot(2,2,3)
        plt.imshow(act_fc3.data.cpu().numpy()[ix,:]>0, aspect='auto', interpolation='nearest')
        plt.subplot(2,2,4)
        plt.imshow(output.data.cpu().numpy()[ix,:], aspect='auto', interpolation='nearest')

    sys.exit() 
