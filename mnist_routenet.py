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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


## DOES IS MAKE SENSE TO USE NN MODULE? BECAUSE OF GATING, WE CAN'T USE A BATCH
## SIZE OF MORE THAN 1.  SO WHAT IS VALUE OF NN MODULE?
class RouteNet(nn.Module):
    def __init__(self, n_input_neurons, idx_input_banks, bank_conn, 
                 idx_output_banks, n_output_neurons, n_neurons_per_hidd_bank=10):
        super(RouteNet, self).__init__()

        # "bank_conn" defines the connectivity of the banks. This is an NxN boolean matrix for 
        # which a True value in the i,j-th entry indictes that bank i is a source of input to
        # bank j. The matrix could define any structure of banks, including for example, a
        # feedforward layered structure or a structure in which all banks are connected.
        n_hidd_banks = bank_conn.shape[0]
        assert (len(bank_conn.shape) == 2), "bank_conn connectivity matrix must have two dimensions of equal size."
        assert (bank_conn.shape[1] == n_hidd_banks), "bank_conn connectivity matrix must have two dimensions of equal size."

        self.n_hidd_banks = n_hidd_banks
        self.bank_conn = bank_conn
        self.idx_input_banks = idx_input_banks
        self.idx_output_banks = idx_output_banks

        # Create all the hidden nn.Linear modules including those for data and those for gates
        for i_source in range(n_hidd_banks):
            for i_target in range(n_hidd_banks):
                if bank_conn[i_source, i_target]:
                    module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
                    setattr(self, module_name, nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank))
                    module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
                    setattr(self, module_name, nn.Linear(n_neurons_per_hidd_bank, 1))

        # Create the connections between inputs and banks that receive inputs
        for i_input_bank in idx_input_banks:
            module_name = 'input_b%0.2d_data' % (i_input_bank)
            setattr(self, module_name, nn.Linear(n_input_neurons, n_neurons_per_hidd_bank))

        # Create the connections between output banks and network output layer
        for i_output_bank in idx_output_banks:
            module_name = 'b%0.2d_output_data' % (i_output_bank)
            setattr(self, module_name, nn.Linear(n_neurons_per_hidd_bank, n_output_neurons))


    def forward(self, x):
        # Definition: A "bank" of neurons is a group of neurons that are not connected to 
        # each other. If two banks are connected, they are fully connected. Inputs to a bank
        # from others banks may be gated on/off in bank-wise fashion. E.g., if Set_i has
        # inputs Set_j and Set_k,
        #    si = a( gji()*Wji*sj + gki()*Wki*sj) ), where gji() is a gating function and a() is an activation function.
        #
        # Gating functions, g(), may be implemented in various neural ways:
        # 1. Source banks have clones which all generate the same output but have different
        #    targets. That target bank has a single gating neuron for each source bank, which
        #    is used to compute the gate function, gji(), and possibly inhibit the dendrites
        #    that take input from that source. Thus only if the gate neuron is "open" do we
        #    need to compute the impact of the relevant source bank. If all input gates are closed
        #    than nothing needs to be done. The default mode for a gate is closed, such that
        #    a "table of open gates" can be iteratively updated and used to determine which banks
        #    need updating and which inputs they should process. This might be done asyncronously.
        #    

        # ASSUMPTION: For now, I'm assuming that the structure is layered and feedforward such that one pass
        # through the connection matrix from low index value to high index value will result in
        # complete processing of the data from input to output.

        bank_data_acts = np.full(self.n_hidd_banks, None)
        n_open_gates = 0
        prob_open_gate = None

        x = x.view(-1, 784)

        # Update activations of all the input banks. These are not gated.
        for i_input_bank in self.idx_input_banks:
            module_name = 'input_b%0.2d_data' % (i_input_bank)
            bank_data_acts[i_input_bank] = F.relu(getattr(self, module_name)(x))
            # TODO: Add activations to total activation energy?

        # Update activations of all the hidden banks. These are gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(bank_conn[:,i_target])[0]

            # Check to see if all source bank activations are None, in which case
            # nothing has to be done.
            if np.all(bank_data_acts[idx_source]==None):
                continue

            # Compute gate values for each of the input banks, and data values if
            # gate is open.
            for i_source in idx_source:
                if bank_data_acts[i_source] is not None:
                    module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
                    gate_act = getattr(self, module_name)(bank_data_acts[i_source])
                    # TODO: Add activations to total activation energy?
                    # TODO: Add gate activations to total gate activation energy.
                    # TODO: Apply hard sigmoid and roll the dice to see if gate is open or closed
                    # Just using above or below zero for now....
                    if gate_act.data[0,0] > 0:
                        n_open_gates += 1
                        module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
                        data_act = getattr(self, module_name)(bank_data_acts[i_source])
                        if bank_data_acts[i_target] is None:
                            bank_data_acts[i_target] = data_act
                        else:
                            bank_data_acts[i_target] += data_act
            if bank_data_acts[i_target] is not None:
                bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
                # TODO: Add activations to total activation energy?

        # Update activations output layer. The output 'bank' is not gated, but it may
        # not have any active inputs, so have to check for that.
        output = None
        if np.all(bank_data_acts[self.idx_output_banks]==None):
            return output, prob_open_gate
        for i_output_bank in self.idx_output_banks:
            if bank_data_acts[i_output_bank] is not None:
                module_name = 'b%0.2d_output_data' % (i_output_bank)
                data_act = getattr(self, module_name)(bank_data_acts[i_output_bank])
                if output is None:
                    output = data_act
                else:
                    output += data_act

        if output is not None:
            output = F.log_softmax(output, dim=1)
            prob_open_gate = n_open_gates / self.n_hidd_banks
        # TODO: Add output activations to total activation energy?

        return output, prob_open_gate

        # p = 0.0
        # x = x.view(-1, 784)
        # act_fc1 = F.relu(self.fc1(x))
        # act_fc1 = F.dropout(act_fc1, p=p, training=self.training)
        # act_fc2 = F.relu(self.fc2(act_fc1))
        # act_fc2 = F.dropout(act_fc2, p=p, training=self.training)
        # act_fc3 = F.relu(self.fc3(act_fc2))
        # act_fc3 = F.dropout(act_fc3, p=p, training=self.training)
        # act_fc4 = self.fc4(act_fc3)
        # return torch.sqrt(act_fc1), torch.sqrt(act_fc2), torch.sqrt(act_fc3), F.log_softmax(act_fc4, dim=1)
        # # return act_fc1, act_fc2, act_fc3, F.log_softmax(act_fc4, dim=1)

def make_conn_matrix(banks_per_layer):
    n_layers = len(banks_per_layer)
    n_banks = np.sum(banks_per_layer)
    bank_conn = np.full((n_banks, n_banks), False)
    idx_banks = []
    cnt = 0
    for i_layer in range(n_layers):
        idx_banks.append(np.arange(cnt, cnt+banks_per_layer[i_layer]))
        cnt += banks_per_layer[i_layer]
    for i_layer in range(n_layers-1):
        bank_conn[np.meshgrid(idx_banks[i_layer], idx_banks[i_layer+1])] = True
    return bank_conn


banks_per_layer = np.asarray([10,10])
bank_conn = make_conn_matrix(banks_per_layer)
model = RouteNet(n_input_neurons=784, \
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
        output = model(data)

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
