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

import routenet as rn

plt.ion()


# TODO: Train/test on CIFAR, and mixed CIFAR-MNIST
# TODO: On mixed CIFAR-MNIST, do we see divergence of routing paths?
# TODO: What fraction of banks are never gated and can thus be removed? Method for adapting network size to fit the data?

# TODO: Does trained hardgate network ever give None as output?
# TODO: Not in near term, but test random ordering of gate and bank updates. Converges to solution?

# TODO: Probabilistic gating.
# TODO: Allow for lambda scheduling: Start with low weight on gating
#       activation and increase with each epoch.
# TODO: Add activation loss. All activations or just gates?

# IDEA: Add loss based on distance between neurons (wiring cost). Helpful in neuromimetic processor (keep processing local)?
# IDEA: Hierarchical routing?  Fractal/hierarchical connectivity patterns/modularity?
# IDEA: Accompanying mechanisms to modulate learning?


# Read in path where raw and processed data are stored
configParser = ConfigParser.RawConfigParser()
configParser.readfp(open(r'config.txt'))

data_section = 'Data Directories'
dirMnistData = configParser.get(data_section, 'dirMnistData')
fullRootFilenameSoftModel = configParser.get(data_section, 'fullRootFilenameSoftModel')
fullRootFilenameHardModel = configParser.get(data_section, 'fullRootFilenameHardModel')

## Get training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
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
                    help='weighting on nll loss. weight on gate activation loss is 1-lambda_nll.')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print('\nUsing CUDA.\n')
else:
    print('\nNot using CUDA.\n')

lambda_nll = args.lambda_nll
lambda_gate = 1 - args.lambda_nll

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


## Define training and testing functions
def train_hardgate(epoch):
    model.train()
    loss_sum = 0.0
    prob_open_gate_sum = 0.0
    cnt = 0
    loss_nll_train_hist = np.asarray([])
    loss_gate_train_hist = np.asarray([])
    loss_total_train_hist = np.asarray([])
    t_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, total_gate_act, prob_open_gate, _ = model.forward_hardgate(data)

        if output is not None:
            # loss_nll = F.nll_loss(output, target) # Use if log_softmax *is* applied in model's forward method
            loss_nll = F.cross_entropy(output, target) # Use if log_softmax *is not* applied in model's forward method
            loss_gate = torch.mean(total_gate_act)
            loss = lambda_nll*loss_nll + lambda_gate*loss_gate

            loss.backward()
            optimizer.step()
            # TODO?: Currently have to use batch size of one. Could we accumulate
            # gradients over a number of samples and then update weights, without
            # using the optimizer? Can't use usual pytorch approach because
            # the constructed graph can be difference for each sample, during
            # training.

            loss_sum = loss_sum + loss.data.cpu().numpy()[0]
            prob_open_gate_sum += prob_open_gate
            cnt += 1

            loss_nll_train_hist = np.append(loss_nll_train_hist, loss_nll.data.cpu().numpy()[0])
            loss_gate_train_hist = np.append(loss_gate_train_hist, loss_gate.data.cpu().numpy()[0])
            loss_total_train_hist = np.append(loss_total_train_hist, loss.data.cpu().numpy()[0])

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%), {:d} valid]\tLoss: {:.6f}\tProb open gate: {:.6f}\t{:.2f} seconds'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.data[0]))
                100. * batch_idx / len(train_loader), cnt, loss_sum/cnt, prob_open_gate_sum/cnt, time.time()-t_start))
            loss_sum = 0.0
            prob_open_gate_sum = 0.0
            cnt = 0
    # return loss_sum/cnt
    return loss_total_train_hist, loss_nll_train_hist, loss_gate_train_hist

def train_softgate(epoch):
    model.train()
    loss_sum = 0.0
    prob_open_gate_sum = 0.0
    cnt = 0
    loss_nll_train_hist = np.asarray([])
    loss_gate_train_hist = np.asarray([])
    loss_total_train_hist = np.asarray([])
    t_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # output, total_gate_act, prob_open_gate = model.forward_softgate(data)
        output, total_gate_act = model.forward_softgate(data)
        prob_open_gate = 0

        # loss_nll = F.nll_loss(output, target) # Use if log_softmax *is* applied in model's forward method
        loss_nll = F.cross_entropy(output, target)  # Use if log_softmax *is not* applied in model's forward method
        loss_gate = torch.mean(total_gate_act)
        loss = lambda_nll*loss_nll + lambda_gate*loss_gate

        loss.backward()
        optimizer.step()

        loss_sum += loss.data.cpu().numpy()[0]
        prob_open_gate_sum += prob_open_gate
        cnt += 1

        loss_nll_train_hist = np.append(loss_nll_train_hist, loss_nll.data.cpu().numpy()[0])
        loss_gate_train_hist = np.append(loss_gate_train_hist, loss_gate.data.cpu().numpy()[0])
        loss_total_train_hist = np.append(loss_total_train_hist, loss.data.cpu().numpy()[0])

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%), Loss: {:.6f}\tProb open gate: {:.6f}\t{:.2f} seconds'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.data[0]))
                100. * batch_idx / len(train_loader), loss_sum/cnt, prob_open_gate_sum/cnt, time.time()-t_start))
            loss_sum = 0.0
            prob_open_gate_sum = 0.0
            cnt = 0
    # return loss_sum/cnt
    return loss_total_train_hist, loss_nll_train_hist, loss_gate_train_hist

def test_hardgate_speed():
    # Just get NN output. No other processing. To assess speed.
    model.eval()
    t_start = time.time()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, _, = model.forward_hardgate(data, return_gate_status=False)
    return time.time() - t_start

def test_hardgate():
    model.eval()
    test_loss_nll = 0
    test_loss_gate = 0
    test_prob_open_gate = 0
    correct = 0
    cnt = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, total_gate_act, prob_open_gate, gate_status = model.forward_hardgate(data)

        # Store target labels and gate status for all samples
        if cnt==0:
            gates_all = gate_status
            targets_all = target.data.cpu().numpy()
        else:
            gates_all = np.append(gates_all, gate_status, axis=0)
            targets_all = np.append(targets_all, target.data.cpu().numpy(), axis=0)

        if output is not None:
            # Accumulate losses, etc.
            # test_loss_nll += F.nll_loss(output, target).data[0] # sum up batch loss
            test_loss_nll += F.cross_entropy(output, target).data[0] # sum up batch loss
            test_loss_gate += torch.mean(total_gate_act).data[0]
            test_prob_open_gate += prob_open_gate

            # Compute accuracy and accumulate
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

            if cnt%1000 == 0:
                print(cnt)
            cnt += 1

    test_loss_nll /= len(test_loader.dataset)
    test_loss_gate /= len(test_loader.dataset)
    test_prob_open_gate /= cnt
    test_loss = lambda_nll*test_loss_nll + lambda_gate*test_loss_gate
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # return test_loss_nll, test_loss_act
    return test_loss, test_loss_nll, test_loss_gate, test_prob_open_gate, acc, gates_all, targets_all

def test_softgate_speed():
    # Just get NN output. No other processing. To assess speed.
    model.eval()
    t_start = time.time()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, _, = model.forward_softgate(data, return_gate_status=False)
    return time.time() - t_start

def test_softgate():
    model.eval()
    test_loss_nll = 0
    test_loss_gate = 0
    test_prob_open_gate = 0
    correct = 0
    cnt = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, total_gate_act, prob_open_gate, gate_status = model.forward_softgate(data, return_gate_status=True)

        # Store target labels and gate status for all samples
        if cnt==0:
            gates_all = gate_status
            targets_all = target.data.cpu().numpy()
        else:
            gates_all = np.append(gates_all, gate_status, axis=0)
            targets_all = np.append(targets_all, target.data.cpu().numpy(), axis=0)

        # Accumulate losses, etc.
        # test_loss_nll += F.nll_loss(output, target).data[0] # sum up batch loss
        test_loss_nll += F.cross_entropy(output, target).data[0]  # sum up batch loss
        test_loss_gate += torch.mean(total_gate_act).data[0]
        test_prob_open_gate += prob_open_gate

        # Compute accuracy and accumulate
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        cnt += 1
        
    test_loss_nll /= len(test_loader.dataset)
    test_loss_gate /= len(test_loader.dataset)
    test_prob_open_gate /= cnt
    test_loss = lambda_nll*test_loss_nll + lambda_gate*test_loss_gate
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, test_loss_nll, test_loss_gate, test_prob_open_gate, acc, gates_all, targets_all


# def test_compare():
#     model.eval()
#     cnt = 0
#     for data, target in test_loader:
#         e12 = None
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output1, total_gate_act1, prob_open_gate1, gate_status1 = model.forward_softgate(data)
#         output2, total_gate_act2, prob_open_gate2, gate_status2 = model.forward_hardgate(data)
#
#         if output2 is None:
#             if not np.all(output1==0):
#                 print('Mismatch.')
#         elif not np.array_equal(output1.data.cpu().numpy(), output2.data.cpu().numpy()):
#             print('Mismatch.')
#
#         if cnt % 1000 == 0:
#             print(cnt)
#         cnt += 1

## Set up DataLoaders
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


## Instantiate network model
banks_per_layer = np.asarray([10, 10])
bank_conn = rn.make_conn_matrix(banks_per_layer)
param_dict = {'n_input_neurons':784,
             'idx_input_banks':np.arange(banks_per_layer[0]),
             'bank_conn':bank_conn,
             'idx_output_banks':np.arange( np.sum(banks_per_layer)-banks_per_layer[-1], np.sum(banks_per_layer) ),
             'n_output_neurons':10,
             'n_neurons_per_hidd_bank':5,
            }
model = rn.RouteNet(**param_dict)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


## Train it, get results on test set, and save the model
loss_total = np.zeros(args.epochs)
loss_nll = np.zeros(args.epochs)
loss_gate = np.zeros(args.epochs)
prob_open_gate = np.zeros(args.epochs)
acc = np.zeros(args.epochs)
t_start = time.time()
loss_nll_best = np.Inf
loss_nll_best_epoch = 0
for epoch in range(0, args.epochs):
    # Train and test
    train_softgate(epoch+1)
    loss_total[epoch], loss_nll[epoch], loss_gate[epoch], prob_open_gate[epoch], acc[epoch], gate_status, target = test_softgate()

    # Save model architecture and params, if it's the best so far on the test set
    if loss_nll[epoch] < loss_nll_best:
        loss_nll_best_epoch = epoch
        loss_nll_best = loss_nll[epoch]
        model.save_model(fullRootFilenameSoftModel)
        print('Best test set accuracy. Saving model.\n')

dur = time.time()-t_start
print('Time = %f, %f sec/epoch' % (dur, dur/args.epochs))



#=======================================
# TODO: Fine-tune using hard gating...
#=======================================
print('\nFINE-TUNING WITH HARD GATING...\n')

## Set up DataLoaders using batch size of 1
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dirMnistData, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dirMnistData, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, **kwargs)

## Compare seed of softgate and hardgate models
# TODO: Forward methods to don't have unnecessary overhead based on requested return items
t_hard = test_hardgate_speed()
print('Hardgate processing duration = %f seconds' % (t_hard))
t_soft = test_softgate_speed()
print('Softgate processing duration = %f seconds' % (t_soft))

sys.exit()

## On test set, compare timing and results of full soft gates and hard gates (not really hard gates, but rather dynamically routed)
# test_compare()
t_start = time.time()
loss_total[epoch], loss_nll[epoch], loss_gate[epoch], prob_open_gate[epoch], acc[epoch], gate_status, target = test_softgate()
print('Softgate processing duration = %f seconds' % (time.time()-t_start))

t_start = time.time()
loss_total[epoch], loss_nll[epoch], loss_gate[epoch], prob_open_gate[epoch], acc[epoch], gate_status, target = test_hardgate()
print('Hardgate processing duration = %f seconds' % (time.time()-t_start))

sys.exit()

# Create new optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr/10, momentum=args.momentum)

## Train it, get results on test set, and save the model
loss_total = np.zeros(args.epochs)
loss_nll = np.zeros(args.epochs)
loss_gate = np.zeros(args.epochs)
prob_open_gate = np.zeros(args.epochs)
acc = np.zeros(args.epochs)
t_start = time.time()
loss_nll_best = np.Inf
loss_nll_best_epoch = 0

# # Adjust the NLL/gate loss weighting
# lambda_nll = 1.0
# lambda_gate = 1 - lambda_nll

for epoch in range(0, args.epochs):
    # Train and test
    train_hardgate(epoch+1)
    loss_total[epoch], loss_nll[epoch], loss_gate[epoch], prob_open_gate[epoch], acc[epoch], gate_status, target = test_softgate()

    # Save model architecture and params, if it's the best so far on the test set
    if loss_nll[epoch] < loss_nll_best:
        loss_nll_best_epoch = epoch
        loss_nll_best = loss_nll[epoch]
        model.save_model(fullRootFilenameHardModel)
        print('Best test set accuracy. Saving model.\n')

dur = time.time()-t_start
print('Time = %f, %f sec/epoch' % (dur, dur/args.epochs))


sys.exit()




fn = 1

## Plot losses for test set
plt.figure(fn)
fn = fn + 1
plt.clf()

f_plot = plt.plot
# f_plot = plt.semilogy
h_subplots = 3
v_subplots = 2
i_subplot = 1

plt.subplot(h_subplots, v_subplots, i_subplot)
i_subplot += 1
f_plot(loss_nll,'bo-')
plt.title('Test set: NLL loss')
plt.xlabel('Epoch')
plt.grid()

plt.subplot(h_subplots, v_subplots, i_subplot)
i_subplot += 1
f_plot(loss_gate, 'o-')
plt.title('Test set: Activation loss')
plt.xlabel('Epoch')
plt.grid()

plt.subplot(h_subplots, v_subplots, i_subplot)
i_subplot += 1
f_plot(loss_total, 'o-')
plt.title('Test set: Total loss')
plt.xlabel('Epoch')
plt.grid()

plt.subplot(h_subplots, v_subplots, i_subplot)
i_subplot += 1
plt.plot(acc, 'o-')
plt.title('Test set: Classification accuracy')
plt.xlabel('Epoch')
plt.grid()

plt.subplot(h_subplots, v_subplots, i_subplot)
i_subplot += 1
plt.plot(100*prob_open_gate, 'o-')
plt.title('Test set: Percentage of gates open')
plt.xlabel('Epoch')
plt.grid()


## Plot the fraction of open gates, grouped by target labels
targets_unique = np.sort(np.unique(target))
plt.figure(fn)
fn = fn + 1
plt.clf()
for i, targ in enumerate(targets_unique):
    idx = np.where(target==targ)[0]
    mn = np.mean(gate_status[idx,:,:], axis=0)
    plt.subplot(3,4,i+1)
    plt.imshow(mn)
    plt.clim(0,1)
    plt.title(targ)

## Plot the fraction of open gates in connectivity map,
## grouped by target labels.
print('Plotting connectivity maps. This may take a minute...')
targets_unique = np.sort(np.unique(target))
plt.figure(fn)
fn = fn + 1
plt.clf()
layer_num = np.zeros((0))
node_num = np.zeros((0))
for i_layer in range(len(banks_per_layer)):
    layer_num = np.append(layer_num, np.full((banks_per_layer[i_layer]), i_layer+1))
    node_num = np.append(node_num, np.arange(banks_per_layer[i_layer])+1)
for i, targ in enumerate(targets_unique):
    idx = np.where(target==targ)[0]
    mn = np.mean(gate_status[idx,:,:], axis=0)
    plt.subplot(3,4,i+1)
    plt.scatter(layer_num, node_num, s=10, facecolors='none', edgecolors='k')
    for i_source in range(np.sum(banks_per_layer)):
        for i_target in range(np.sum(banks_per_layer)):
            alpha = mn[i_source, i_target]
            plt.plot((layer_num[i_source], layer_num[i_target]), (node_num[i_source], node_num[i_target]), 'k-', alpha=alpha)
    plt.title(targ)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
print('Done.')

sys.exit()

weights2 = []
grads2 = []
for param in model2.parameters():
    weights2.append(param.data)
    grads2.append(param.grad)

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
