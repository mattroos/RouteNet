# routnet.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import sys
from scipy.linalg import toeplitz

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import random

from batchscale import BatchScale

import pdb

## Function to deal with annoying discrepency between pytorch versions,
## some of which return loss values as scalar tensors, others as
## array tensors of size 1.
def item(x):
    n = len(x.size())
    if n==0:
        return x.item()
    elif n==1:
        return x.data.cpu().numpy()[0]
    else:
        print('routenet.item(): Could not convert presumed single-element tensor as scalar.')


def make_conn_matrix_ff_full(banks_per_layer):
    # A fully-connected feed-forward network.
    # banks_per_layer is a list or array
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

def make_conn_matrix_ff_part(n_layers, n_banks_per_layer, n_fan_out):
    # n_banks_per_layer is a scalar.  All layers have the same number of banks.
    # n_fan_out is the number of output banks projected to to by an input bank.
    # Connections are spatially arranged such that the n_fan_out banks are
    # as close as possible to the input bank (vertically, in a FF network
    # that projects rightward in a schematic), centered at the same vertical
    # level as the source bank.
    
    # Build connectivity sub-matrix for single pair of adjacent layers
    row = np.full(n_banks_per_layer, False)
    col = np.full(n_banks_per_layer, False)
    n_fan_down = n_fan_out/2
    n_fan_up = n_fan_out - n_fan_down - 1
    row[0:n_fan_down+1] = True
    col[0:n_fan_up+1] = True
    sub_conn = toeplitz(row, col)

    # Get locations for submatrices
    i_upper = np.arange(n_layers-1) * n_banks_per_layer
    i_left = np.arange(1,n_layers) * n_banks_per_layer

    # Create empty matrix and insert submatrices
    n_banks = n_layers * n_banks_per_layer
    bank_conn = np.full((n_banks, n_banks), False)
    for iu, il in zip(i_upper, i_left):
        bank_conn[iu:iu+n_banks_per_layer, il:il+n_banks_per_layer] = sub_conn

    return bank_conn


def make_conn_matrix_ff_part_2d(n_layers, n_banks_per_layer_per_dim, n_fan_out_per_dim):
    # Create connectivity matrix for layered feed-forward hierarchy
    # with 2-D bank arrangement in each layer and limited fan-out. Assumes
    # same number of banks in each layer, with a square (k x k) arrangement.
    #
    # n_banks_per_layer_per_dim is a scalar. All layers have the same number of banks.
    # n_fan_out_per_dim is the number of output banks projected to by an input bank
    # in each dimension.
    #
    # Connections are spatially arranged such that the n_fan_out_per_dim**2 banks are
    # as close as possible to the input bank, centered at the same (i,j) position
    # as the source bank.
    
    n_banks_per_layer = n_banks_per_layer_per_dim**2
    assert np.mod(n_fan_out_per_dim,2)==1, 'make_conn_matrix_ff_part_2d(): n_fan_out_per_dim must be an odd scalar'
    fh = (int(n_fan_out_per_dim)-1)/2   # length of one-side ("half") of fan-out

    # Build connectivity sub-matrix for single pair of adjacent layers.
    # No doubt there is more efficient code for this.
    sub_conn = np.full((n_banks_per_layer, n_banks_per_layer), False)
    a = np.expand_dims(np.arange(-fh,fh+1), 1)
    for x_targ in range(n_banks_per_layer_per_dim):
        for y_targ in range(n_banks_per_layer_per_dim):
            i_targ = x_targ * n_banks_per_layer_per_dim + y_targ
            x_src = x_targ + a
            y_src = y_targ + a
            x_src, y_src = np.meshgrid(x_src, y_src)
            x_src = x_src.flatten()
            y_src = y_src.flatten()
            idx = np.where((x_src>=0) & (y_src>=0) & (x_src<n_banks_per_layer_per_dim) & (y_src<n_banks_per_layer_per_dim))[0]
            i_src = x_src[idx]*n_banks_per_layer_per_dim + y_src[idx]
            sub_conn[i_src, i_targ] = True

    # Get locations for submatrices in larger matrix
    i_upper = np.arange(n_layers-1) * n_banks_per_layer
    i_left = np.arange(1,n_layers) * n_banks_per_layer

    # Create empty matrix and insert submatrices
    n_banks = n_layers * n_banks_per_layer
    bank_conn = np.full((n_banks, n_banks), False)
    for iu, il in zip(i_upper, i_left):
        bank_conn[iu:iu+n_banks_per_layer, il:il+n_banks_per_layer] = sub_conn

    return bank_conn


class RouteNet(nn.Module):
    def __init__(self, n_input_neurons, idx_input_banks, bank_conn, 
                 idx_output_banks, n_output_neurons, n_neurons_per_hidd_bank=10):
        super(RouteNet, self).__init__()

        self.n_input_neurons = n_input_neurons
        self.idx_input_banks = idx_input_banks
        self.bank_conn = bank_conn
        self.idx_output_banks = idx_output_banks
        self.n_output_neurons = n_output_neurons
        self.n_neurons_per_hidd_bank = n_neurons_per_hidd_bank

        # "bank_conn" defines the connectivity of the banks. This is an NxN boolean matrix for 
        # which a True value in the i,j-th entry indictes that bank i is a source of input to
        # bank j. The matrix could define any structure of banks, including for example, a
        # feedforward layered structure or a structure in which all banks are connected.
        n_hidd_banks = bank_conn.shape[0]
        assert (len(bank_conn.shape) == 2), "bank_conn connectivity matrix must have two dimensions of equal size."
        assert (bank_conn.shape[1] == n_hidd_banks), "bank_conn connectivity matrix must have two dimensions of equal size."

        self.n_hidd_banks = n_hidd_banks
        self.n_bank_conn = np.sum(bank_conn)
        self.prob_dropout_data = 0.0
        self.prob_dropout_gate = 0.0

        # Create all the hidden nn.Linear modules including those for data and those for gates.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        # Use dropout?  Apply same single dropout to each source?  Each source/target combo?
        # Each source/target combo and each source/gate combo?
        self.hidden2hidden_gate = nn.ModuleList()
        self.hidden2hidden_data = nn.ModuleList()
        self.hidden2hidden_gate_dropout = nn.ModuleList()
        self.hidden2hidden_data_dropout = nn.ModuleList()
        self.hidden_batch_norm = nn.ModuleList()
        for i_source in range(n_hidd_banks):
            self.hidden2hidden_gate.append(nn.ModuleList())
            self.hidden2hidden_data.append(nn.ModuleList())
            self.hidden2hidden_gate_dropout.append(nn.ModuleList())
            self.hidden2hidden_data_dropout.append(nn.ModuleList())
            self.hidden_batch_norm.append(nn.BatchNorm1d(self.n_neurons_per_hidd_bank, affine=False))
            # self.hidden_batch_norm.append(BatchScale1d(self.n_neurons_per_hidd_bank, linear=True)
            for i_target in range(n_hidd_banks):
                if bank_conn[i_source, i_target]:
                    # 6/25/18: New thought... To have functional equivalence between hard and
                    # soft gating, we don't necessarily need the gate layers to be unbiased. What we
                    # need is for either the data layers or the gate layers to be unbiased. In 
                    # that case, if the source bank is inactive (all zeros), then the gated-multiplied
                    # weighted sum at the target bank will be zeros either because the gate value
                    # is zero (unbiased gate layer) or the weighted data sum will be zero (unbiased
                    # data layer).
                    # For unknown reasons at this time, the model learns as expected when the gate
                    # layers are biased by the data layers are not. Not clear why it's not working
                    # when the reverse is true.
                    self.hidden2hidden_gate[i_source].append(nn.Linear(n_neurons_per_hidd_bank, 1, bias=False))
                    self.hidden2hidden_data[i_source].append(nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank, bias=True))
                    # self.hidden2hidden_gate[i_source].append(nn.Linear(n_neurons_per_hidd_bank, 1, bias=True))
                    # self.hidden2hidden_data[i_source].append(nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank, bias=False))
                    self.hidden2hidden_gate_dropout[i_source].append(nn.Dropout(p=self.prob_dropout_gate))
                    self.hidden2hidden_data_dropout[i_source].append(nn.Dropout(p=self.prob_dropout_data))
                else:
                    self.hidden2hidden_gate[i_source].append(None)
                    self.hidden2hidden_data[i_source].append(None)
                    self.hidden2hidden_gate_dropout[i_source].append(None)
                    self.hidden2hidden_data_dropout[i_source].append(None)

        # Create the connections between inputs and banks that receive inputs
        self.input_batch_norm = nn.BatchNorm1d(self.n_input_neurons)
        self.input2hidden = nn.ModuleList()
        for i_input_bank in range(n_hidd_banks):
            self.input2hidden.append(None)
        for i_input_bank in idx_input_banks:
            # TODO: Should layers between inputs and receiving banks have a bias or not?
            self.input2hidden[i_input_bank] = nn.Linear(n_input_neurons, n_neurons_per_hidd_bank)

        # Create the connections between output banks and network output layer.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        self.hidden2output = nn.ModuleList()
        for i_output_bank in range(n_hidd_banks):
            self.hidden2output.append(None)
        for i_output_bank in idx_output_banks:
            self.hidden2output[i_output_bank] = nn.Linear(n_neurons_per_hidd_bank, n_output_neurons, bias=False)

    @classmethod
    def init_from_files(cls, model_base_filename):
        # Load model metaparameters, instantiate a model with that architecture,
        # load model weights, and set the model weights.
        print('\tInitializing model architecture...')
        param_dict = np.load('%s.npy' % (model_base_filename)).item()
        net = cls(**param_dict)
        # if b_use_cuda:
        #     net = net.cuda()
        print('\tLoading model parameter values...')
        #net.load_state_dict(torch.load('%s.tch' % (model_base_filename)))
        x = torch.load('%s.tch' % (model_base_filename))
        net.load_state_dict(x)
        return net

    def forward_softgate(self, x, return_gate_status=False, b_use_cuda=False):
        # Unlike the main forward() method, this one uses soft gates thus
        # allowing batches to be used in training. The notion is that this
        # could be used for fast pre-training, and then forward() used for
        # final training with hard gating.
        b_batch_norm = True

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension

        bank_data_acts = np.full(self.n_hidd_banks, None)
        n_open_gates = 0
        output = None
        total_gate_act = 0

        if return_gate_status:
            gate_status = np.full((batch_size,) + self.bank_conn.shape, False)
            # gate_status_value = np.full((batch_size,) + self.bank_conn.shape, -100.0)

        # Batch norm the inputs.
        if b_batch_norm:
            x = self.input_batch_norm(x)

        # Update activations of all the input banks. These are not gated.
        for i_input_bank in self.idx_input_banks:
            bank_data_acts[i_input_bank] = F.relu(self.input2hidden[i_input_bank](x))

        # Update activations of all the hidden banks. These are soft gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Compute gate values for each of the input banks, and multiply
            # by the incoming activations.
            for idx, i_source in enumerate(idx_source):
                dropout_act = self.hidden2hidden_gate_dropout[i_source][i_target](bank_data_acts[i_source])
                gate_act = self.hidden2hidden_gate[i_source][i_target](dropout_act)
                
                ## Apply hard sigmoid or RELU
                # gate_act = F.relu(gate_act)
                gate_act = F.hardtanh(gate_act, 0.0, 1.0)

                if args.neg_gate_loss:
                    total_gate_act -= gate_act
                else:
                    total_gate_act += gate_act

                if return_gate_status:
                    gate_status[:, i_source, i_target] = gate_act.data.cpu().numpy()[:,0] > 0
                    #gate_status_value[:, i_source, i_target] = gate_act.data.cpu().numpy()[:,0]
                    z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
                    n_open_gates += np.sum(z)

                dropout_act = self.hidden2hidden_data_dropout[i_source][i_target](bank_data_acts[i_source])
                data_act = self.hidden2hidden_data[i_source][i_target](dropout_act)

                if bank_data_acts[i_target] is None:
                    bank_data_acts[i_target] = gate_act * data_act
                else:
                    bank_data_acts[i_target] += gate_act * data_act

            bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
            if b_batch_norm:
                bank_data_acts[i_target] = self.hidden_batch_norm[i_target](bank_data_acts[i_target])

        if return_gate_status:
            prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size)

        # Update activations of the output layer. The output banks are not gated.
        for i_output_bank in self.idx_output_banks:
            data_act = self.hidden2output[i_output_bank](bank_data_acts[i_output_bank])

            if output is None:
                output = data_act
            else:
                output += data_act

        total_gate_act /= self.n_bank_conn  # average per connection

        if return_gate_status:
            #return output, total_gate_act, prob_open_gate, gate_status, gate_status_value
            return output, total_gate_act, prob_open_gate, gate_status
        else:
            return output, total_gate_act

    def save_model(self, model_base_filename):
        # Just saving the model, not the optimizer state. To stop and 
        # resume training, optimizer state needs to be saved as well.
        # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610
        param_dict = {
            'n_input_neurons':self.n_input_neurons,
            'idx_input_banks':self.idx_input_banks,
            'bank_conn':self.bank_conn,
            'idx_output_banks':self.idx_output_banks,
            'n_output_neurons':self.n_output_neurons,
            'n_neurons_per_hidd_bank':self.n_neurons_per_hidd_bank
        }
        torch.save(self.state_dict(), '%s.tch' % (model_base_filename))
        np.save('%s.npy' % (model_base_filename), param_dict)


class RouteNetOneToOneOutputGroupedInputs(nn.Module):
    # RouteNet that has one output bank per output node
    # and there are multiple groups of inputs, each of
    # which is connected to a single input bank. Each
    # input group must have the same number of neurons.
    # NOTE: In this class, the length of idx_input_banks
    # defines the expected number of input groups, that is,
    # the length of the list, x, in the forward methods.
    def __init__(self, n_neurons_per_input_group, idx_input_banks, bank_conn, 
                 idx_output_banks, n_neurons_per_hidd_bank=10):
        super(RouteNetOneToOneOutputGroupedInputs, self).__init__()

        self.n_neurons_per_input_group = n_neurons_per_input_group
        self.idx_input_banks = idx_input_banks
        self.bank_conn = bank_conn
        self.idx_output_banks = idx_output_banks
        self.n_output_neurons = len(idx_output_banks)
        self.n_neurons_per_hidd_bank = n_neurons_per_hidd_bank

        # "bank_conn" defines the connectivity of the banks. This is an NxN boolean matrix for 
        # which a True value in the i,j-th entry indictes that bank i is a source of input to
        # bank j. The matrix could define any structure of banks, including for example, a
        # feedforward layered structure or a structure in which all banks are connected.
        n_hidd_banks = bank_conn.shape[0]
        assert (len(bank_conn.shape) == 2), "bank_conn connectivity matrix must have two dimensions of equal size."
        assert (bank_conn.shape[1] == n_hidd_banks), "bank_conn connectivity matrix must have two dimensions of equal size."

        n_input_groups = len(idx_input_banks)

        self.n_hidd_banks = n_hidd_banks
        self.n_bank_conn = np.sum(bank_conn)
        self.n_input_groups = n_input_groups
        self.prob_dropout_data = 0.0
        self.prob_dropout_gate = 0.0

        # Create all the hidden nn.Linear modules including those for data and those for gates.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        # Use dropout?  Apply same single dropout to each source?  Each source/target combo?
        # Each source/target combo and each source/gate combo?
        self.hidden2hidden_gate = nn.ModuleList()
        self.hidden2hidden_data = nn.ModuleList()
        self.hidden2hidden_gate_dropout = nn.ModuleList()
        self.hidden2hidden_data_dropout = nn.ModuleList()
        self.hidden_batch_scale = nn.ModuleList()
        for i_source in range(n_hidd_banks):
            self.hidden2hidden_gate.append(nn.ModuleList())
            self.hidden2hidden_data.append(nn.ModuleList())
            self.hidden2hidden_gate_dropout.append(nn.ModuleList())
            self.hidden2hidden_data_dropout.append(nn.ModuleList())
            # self.hidden_batch_scale.append(nn.BatchNorm1d(self.n_neurons_per_hidd_bank, affine=True))  # actually linear, not affine, since no bias
            self.hidden_batch_scale.append(BatchScale(self.n_neurons_per_hidd_bank)) # Can't use BatchNorm: batch shifting is incompatible with hard gating.
            for i_target in range(n_hidd_banks):
                if bank_conn[i_source, i_target]:
                    # To have functional equivalence between hard and soft gating, we don't
                    # necessarily need the gate layers to be unbiased. What we need is for
                    # either the data layers or the gate layers to be unbiased. In that
                    # case, if the source bank is inactive (all zeros), then the
                    # gated-multiplied weighted sum at the target bank will be zeros either
                    # because the gate value is zero (unbiased gate layer) or the weighted
                    # data sum will be zero (unbiased data layer).
                    # For unknown reasons at this time, the model learns as desired when the
                    # gate layers are biased but the data layers are not. Not clear why it
                    # doesn't learn as well when the reverse is true.
                    self.hidden2hidden_gate[i_source].append(nn.Linear(n_neurons_per_hidd_bank, 1, bias=True))
                    self.hidden2hidden_data[i_source].append(nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank, bias=False))
                    self.hidden2hidden_gate_dropout[i_source].append(nn.Dropout(p=self.prob_dropout_gate))
                    self.hidden2hidden_data_dropout[i_source].append(nn.Dropout(p=self.prob_dropout_data))
                else:
                    self.hidden2hidden_gate[i_source].append(None)
                    self.hidden2hidden_data[i_source].append(None)
                    self.hidden2hidden_gate_dropout[i_source].append(None)
                    self.hidden2hidden_data_dropout[i_source].append(None)

        # Create the connections between inputs and banks that receive inputs.
        # Not using gates on the inputs, so can use BatchNorm rather than BatchScale.
        self.input_batch_norm = nn.ModuleList()
        self.input2hidden_data = nn.ModuleList()
        for i_input_group in range(n_input_groups):
            # self.input_batch_norm.append(nn.BatchNorm1d(self.n_neurons_per_input_group))
            # self.input2hidden_data.append(nn.Linear(self.n_neurons_per_input_group, n_neurons_per_hidd_bank, bias=True))
            self.input_batch_norm.append(BatchScale(self.n_neurons_per_input_group))
            self.input2hidden_data.append(nn.Linear(self.n_neurons_per_input_group, n_neurons_per_hidd_bank, bias=False))

        # Create the connections between output banks and network output layer.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        self.hidden2output = nn.ModuleList()
        for i_output_bank in range(n_hidd_banks):
            self.hidden2output.append(None)
        for i_output_bank in idx_output_banks:
            self.hidden2output[i_output_bank] = nn.Linear(n_neurons_per_hidd_bank, 1, bias=False)

    def freeze_data_params(self):
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze gate parameters
        for i_source in range(self.n_hidd_banks):
            for i_target in range(self.n_hidd_banks):
                if self.bank_conn[i_source, i_target]:
                    for param in self.hidden2hidden_gate[i_source][i_target].parameters():
                        param.requires_grad = True

    def freeze_gate_params(self):
        for i_source in range(self.n_hidd_banks):
            for i_target in range(self.n_hidd_banks):
                if self.bank_conn[i_source, i_target]:
                    for param in self.hidden2hidden_gate[i_source][i_target].parameters():
                        param.requires_grad = False

    def unfreeze_all_params(self):
        for param in self.parameters():
            param.requires_grad = True

    def init_gate_bias(self, bias=0):
        for i_source in range(self.n_hidd_banks):
            for i_target in range(self.n_hidd_banks):
                if self.bank_conn[i_source, i_target]:
                    self.hidden2hidden_gate[i_source, i_target].bias.data.fill_(bias)

    @classmethod
    def init_from_files(cls, model_base_filename):
        # Load model metaparameters, instantiate a model with that architecture,
        # load model weights, and set the model weights.
        print('\tInitializing model architecture...')
        param_dict = np.load('%s.npy' % (model_base_filename)).item()
        net = cls(**param_dict)
        # if b_use_cuda:
        #     net = net.cuda()
        print('\tLoading model parameter values...')
        #net.load_state_dict(torch.load('%s.tch' % (model_base_filename)))
        x = torch.load('%s.tch' % (model_base_filename))
        net.load_state_dict(x) # this is very slow. too many torch modules.
        return net

    def forward_softgate(self, x, return_gate_status=False, b_batch_norm=False, b_use_cuda=False, b_no_gates=False, b_neg_gate_loss=False):
        # Unlike the main forward() method, this one uses soft gates thus
        # allowing batches to be used in training. The notion is that this
        # could be used for fast pre-training, and then forward_hardgate()
        # used for inference on single examples/samples, or possibly for final,
        # fine-tuning training with hard gating.

        batch_size = x[0].size()[0]
        for i in range(self.n_input_groups):
            x[i] = x[i].view(batch_size, -1)  # Flatten across all dimensions except batch dimension

        bank_data_acts = np.full(self.n_hidd_banks, None)
        n_open_gates = 0
        total_gate_act = Variable(torch.zeros(batch_size,1))
        output = Variable(torch.zeros(batch_size,self.n_output_neurons))
        if b_use_cuda:
            total_gate_act = total_gate_act.cuda()
            output = output.cuda()

        if return_gate_status:
            gate_status = np.full((batch_size,) + self.bank_conn.shape, False)
            #gate_status_value = np.full((batch_size,) + self.bank_conn.shape, -100.0)

        # Batch norm the inputs.
        if b_batch_norm:
            for i_input_group in range(self.n_input_groups):
                x[i_input_group] = self.input_batch_norm[i_input_group](x[i_input_group])

        # Update activations of all the input banks. These are not gated.
        for i_input_group, idx_input_bank in enumerate(self.idx_input_banks):
            # bank_data_acts[idx_input_bank] = F.relu(self.input2hidden_data[i_input_group](x[i_input_group]))
            bank_data_acts[idx_input_bank] = self.input2hidden_data[i_input_group](x[i_input_group])

        # Update activations of all the hidden banks. These are soft gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Compute gate values for each of the input banks, and multiply
            # by the incoming activations.
            for idx, i_source in enumerate(idx_source):
                dropout_act = self.hidden2hidden_gate_dropout[i_source][i_target](bank_data_acts[i_source])
                gate_act = self.hidden2hidden_gate[i_source][i_target](dropout_act)
                
                z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
                # z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int) & \
                #     np.any(bank_data_acts[i_source].data, axis=1).flatten().astype(np.int)
                n_open_gates += np.sum(z)

                ## Apply hard sigmoid or RELU
                # gate_act = F.relu(gate_act)
                # gate_act = (gate_act - 1.0)**2
                # gate_act = F.hardtanh(gate_act, -100.0, 1.0)
                gate_act = F.hardtanh(gate_act, 0.0, 1.0)

                # if not b_no_gates:
                # Compute gate_loss even if gates aren't applied to the data.
                # Can set loss weighting factor such that gate loss isn't
                # relevent, if wanted (or the reverse!).
                if b_neg_gate_loss:
                    total_gate_act -= gate_act
                else:
                    total_gate_act += gate_act

                if return_gate_status:
                    # Gate status is set to True (open) only if both the gate node is
                    # greater than zero and one or more activations from the source
                    # bank are non-zero.  I.e., if the gate is open but all the data
                    # inputs are zeros, this is functionally the same as a closed gate.
                    gate_status[:, i_source, i_target] = (gate_act.data.cpu().numpy()[:,0] > 0) #& \
                                                         #np.any(bank_data_acts[i_source].data, axis=1)
                    #gate_status_value[:, i_source, i_target] = gate_act.data.cpu().numpy()[:,0]

                dropout_act = self.hidden2hidden_data_dropout[i_source][i_target](bank_data_acts[i_source])
                data_act = self.hidden2hidden_data[i_source][i_target](dropout_act)

                if bank_data_acts[i_target] is None:
                    if b_no_gates:
                        bank_data_acts[i_target] = data_act
                    else:
                        bank_data_acts[i_target] = gate_act * data_act
                else:
                    if b_no_gates:
                        bank_data_acts[i_target] += data_act
                    else:
                        bank_data_acts[i_target] += gate_act * data_act

            bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
            if b_batch_norm:
                bank_data_acts[i_target] = self.hidden_batch_scale[i_target](bank_data_acts[i_target])

        prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size)

        # Update activations of the output layer. The output banks are not gated.
        for i_output_neuron, i_output_bank in enumerate(self.idx_output_banks):
            data_act = self.hidden2output[i_output_bank](bank_data_acts[i_output_bank])
            # output[:,i_output_neuron] = data_act
            output[:,i_output_neuron] = data_act[:,0]
            # if output is None:
            #     output = data_act
            # else:
            #     output += data_act

        # Should we gate the one-to-one outputs?  Just trying RELU for now...
        # output = F.relu(output)

        total_gate_act /= self.n_bank_conn  # average per connection

        if return_gate_status:
            return output, total_gate_act, prob_open_gate, gate_status
            #return output, total_gate_act, prob_open_gate, gate_status, gate_status_value
        else:
            return output, total_gate_act, prob_open_gate

    def save_model(self, model_base_filename):
        # Just saving the model, not the optimizer state. To stop and 
        # resume training, optimizer state needs to be saved as well.
        # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610
        param_dict = {
            'n_neurons_per_input_group':self.n_neurons_per_input_group,
            'idx_input_banks':self.idx_input_banks,
            'bank_conn':self.bank_conn,
            'idx_output_banks':self.idx_output_banks,
            'n_neurons_per_hidd_bank':self.n_neurons_per_hidd_bank
        }
        torch.save(self.state_dict(), '%s.tch' % (model_base_filename))
        np.save('%s.npy' % (model_base_filename), param_dict)


class RouteNetOneToOneOutput(nn.Module):
    # RouteNet that has one output bank per output node
    def __init__(self, n_input_neurons, idx_input_banks, bank_conn, 
                 idx_output_banks, n_neurons_per_hidd_bank=10):
        super(RouteNetOneToOneOutput, self).__init__()

        self.n_input_neurons = n_input_neurons
        self.idx_input_banks = idx_input_banks
        self.bank_conn = bank_conn
        self.idx_output_banks = idx_output_banks
        self.n_output_neurons = len(idx_output_banks)
        self.n_neurons_per_hidd_bank = n_neurons_per_hidd_bank

        # "bank_conn" defines the connectivity of the banks. This is an NxN boolean matrix for 
        # which a True value in the i,j-th entry indictes that bank i is a source of input to
        # bank j. The matrix could define any structure of banks, including for example, a
        # feedforward layered structure or a structure in which all banks are connected.
        n_hidd_banks = bank_conn.shape[0]
        assert (len(bank_conn.shape) == 2), "bank_conn connectivity matrix must have two dimensions of equal size."
        assert (bank_conn.shape[1] == n_hidd_banks), "bank_conn connectivity matrix must have two dimensions of equal size."

        self.n_hidd_banks = n_hidd_banks
        self.n_bank_conn = np.sum(bank_conn)
        self.prob_dropout_data = 0.0
        self.prob_dropout_gate = 0.0

        # Create all the hidden nn.Linear modules including those for data and those for gates.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        # Use dropout?  Apply same single dropout to each source?  Each source/target combo?
        # Each source/target combo and each source/gate combo?
        self.hidden2hidden_gate = nn.ModuleList()
        self.hidden2hidden_data = nn.ModuleList()
        self.hidden2hidden_gate_dropout = nn.ModuleList()
        self.hidden2hidden_data_dropout = nn.ModuleList()
        self.hidden_batch_scale = nn.ModuleList()
        for i_source in range(n_hidd_banks):
            self.hidden2hidden_gate.append(nn.ModuleList())
            self.hidden2hidden_data.append(nn.ModuleList())
            self.hidden2hidden_gate_dropout.append(nn.ModuleList())
            self.hidden2hidden_data_dropout.append(nn.ModuleList())
            # self.hidden_batch_scale.append(nn.BatchNorm1d(self.n_neurons_per_hidd_bank, affine=True))  # actually linear, not affine, since no bias
            self.hidden_batch_scale.append(BatchScale(self.n_neurons_per_hidd_bank)) # Can't use BatchNorm: batch shifting is incompatible with hard gating.
            for i_target in range(n_hidd_banks):
                if bank_conn[i_source, i_target]:
                    # To have functional equivalence between hard and soft gating, we don't
                    # necessarily need the gate layers to be unbiased. What we need is for
                    # either the data layers or the gate layers to be unbiased. In that
                    # case, if the source bank is inactive (all zeros), then the
                    # gated-multiplied weighted sum at the target bank will be zeros either
                    # because the gate value is zero (unbiased gate layer) or the weighted
                    # data sum will be zero (unbiased data layer).
                    # For unknown reasons at this time, the model learns as desired when the
                    # gate layers are biased but the data layers are not. Not clear why it
                    # doesn't learn as well when the reverse is true.

                    # # Works, but cheating. Data nodes could fire even if gated off. Use bias <=0 only?
                    # self.hidden2hidden_gate[i_source].append(nn.Linear(n_neurons_per_hidd_bank, 1, bias=False))
                    # self.hidden2hidden_data[i_source].append(nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank, bias=True))

                    # Works okay?
                    self.hidden2hidden_gate[i_source].append(nn.Linear(n_neurons_per_hidd_bank, 1, bias=True))
                    self.hidden2hidden_data[i_source].append(nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank, bias=False))

                    # # Works okay? And is best for implementation as hard gating.
                    # self.hidden2hidden_gate[i_source].append(nn.Linear(n_neurons_per_hidd_bank, 1, bias=False))
                    # self.hidden2hidden_data[i_source].append(nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank, bias=False))

                    self.hidden2hidden_gate_dropout[i_source].append(nn.Dropout(p=self.prob_dropout_gate))
                    self.hidden2hidden_data_dropout[i_source].append(nn.Dropout(p=self.prob_dropout_data))
                else:
                    self.hidden2hidden_gate[i_source].append(None)
                    self.hidden2hidden_data[i_source].append(None)
                    self.hidden2hidden_gate_dropout[i_source].append(None)
                    self.hidden2hidden_data_dropout[i_source].append(None)

        # Create the connections between inputs and banks that receive inputs.
        # Not using gates on the inputs, so can use BatchNorm rather than BatchScale.
        self.input_batch_norm = nn.BatchNorm1d(self.n_input_neurons)
        self.input2hidden = nn.ModuleList()
        for i_input_bank in range(n_hidd_banks):
            self.input2hidden.append(None)
        for i_input_bank in idx_input_banks:
            self.input2hidden[i_input_bank] = nn.Linear(n_input_neurons, n_neurons_per_hidd_bank, bias=True)

        # Create the connections between output banks and network output layer.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        self.hidden2output = nn.ModuleList()
        for i_output_bank in range(n_hidd_banks):
            self.hidden2output.append(None)
        for i_output_bank in idx_output_banks:
            self.hidden2output[i_output_bank] = nn.Linear(n_neurons_per_hidd_bank, 1, bias=False)

    @classmethod
    def init_from_files(cls, model_base_filename):
        # Load model metaparameters, instantiate a model with that architecture,
        # load model weights, and set the model weights.
        param_dict = np.load('%s.npy' % (model_base_filename)).item()
        net = cls(**param_dict)
        # if b_use_cuda:
        #     net = net.cuda()
        net.load_state_dict(torch.load('%s.tch' % (model_base_filename)))
        return net

    def forward_softgate(self, x, return_gate_status=False, b_batch_norm=False, b_use_cuda=False, b_no_gates=False, b_neg_gate_loss=False):
        # Unlike the main forward() method, this one uses soft gates thus
        # allowing batches to be used in training. The notion is that this
        # could be used for fast pre-training, and then forward_hardgate()
        # used for inference on single examples/samples, or possibly for final,
        # fine-tuning training with hard gating.

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension

        bank_data_acts = np.full(self.n_hidd_banks, None)
        n_open_gates = 0
        total_gate_act = Variable(torch.zeros(batch_size,1))
        output = Variable(torch.zeros(batch_size,self.n_output_neurons))
        if b_use_cuda:
            total_gate_act = total_gate_act.cuda()
            output = output.cuda()

        if return_gate_status:
            gate_status = np.full((batch_size,) + self.bank_conn.shape, False)

        # Batch norm the inputs.
        if b_batch_norm:
            x = self.input_batch_norm(x)

        # Update activations of all the input banks. These are not gated. Do not apply
        # activation non-linearity here, as that is done in the loop below.
        for i_input_bank in self.idx_input_banks:
            bank_data_acts[i_input_bank] = self.input2hidden[i_input_bank](x)

        # Update activations of all the hidden banks. These are soft gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Compute gate values for each of the input banks, and multiply
            # by the incoming activations.
            for idx, i_source in enumerate(idx_source):
                dropout_act = self.hidden2hidden_gate_dropout[i_source][i_target](bank_data_acts[i_source])
                gate_act = self.hidden2hidden_gate[i_source][i_target](dropout_act)
                
                ## Apply hard sigmoid or RELU
                # gate_act = F.relu(gate_act)
                gate_act = F.hardtanh(gate_act, 0.0, 1.0)

                if not b_no_gates:
                    if b_neg_gate_loss:
                        total_gate_act -= gate_act
                    else:
                        total_gate_act += gate_act

                if return_gate_status:
                    # Gate status is set to True (open) only if both the gate node is
                    # greater than zero and one or more activations from the source
                    # bank are non-zero.  I.e., if the gate is open but all the data
                    # inputs are zeros, this is functionally the same as a closed gate.
                    # gate_status[:, i_source, i_target] = (gate_act.data.cpu().numpy()[:,0] > 0) & \
                    #                                      np.any(bank_data_acts[i_source].data, axis=1)
                    gate_status[:, i_source, i_target] = (gate_act.data.cpu().numpy()[:,0] > 0)
                    # if not gate_status[:, i_source, i_target]:
                    #     pdb.set_trace()

                z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
                n_open_gates += np.sum(z)

                dropout_act = self.hidden2hidden_data_dropout[i_source][i_target](bank_data_acts[i_source])
                data_act = self.hidden2hidden_data[i_source][i_target](dropout_act)

                if bank_data_acts[i_target] is None:
                    if b_no_gates:
                        bank_data_acts[i_target] = data_act
                    else:
                        bank_data_acts[i_target] = gate_act * data_act
                else:
                    if b_no_gates:
                        bank_data_acts[i_target] += data_act
                    else:
                        bank_data_acts[i_target] += gate_act * data_act

            bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
            if b_batch_norm:
                bank_data_acts[i_target] = self.hidden_batch_scale[i_target](bank_data_acts[i_target])

        prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size)

        # Update activations of the output layer. The output banks are not gated.
        for i_output_neuron, i_output_bank in enumerate(self.idx_output_banks):
            data_act = self.hidden2output[i_output_bank](bank_data_acts[i_output_bank])
            output[:,i_output_neuron] = data_act

        # Should we gate the one-to-one outputs?  Just trying RELU for now...
        output = F.relu(output)

        total_gate_act /= self.n_bank_conn  # average per connection

        if return_gate_status:
            return output, total_gate_act, prob_open_gate, gate_status
        else:
            return output, total_gate_act, prob_open_gate

    def forward_hardgate(self, x, return_gate_status=False, b_batch_norm=False, b_use_cuda=False, b_neg_gate_loss=False):
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

        # ASSUMPTION: For now, I'm assuming that the structure is layered and feedforward such that one pass
        # through the connection matrix from low index value to high index value will result in
        # complete processing of the data from input to output.

        batch_size = x.size()[0]
        assert batch_size==1, 'batch_size must be 1 for forward_hardgate().'
        x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension

        bank_data_acts = np.full(self.n_hidd_banks, None)
        n_open_gates = 0
        prob_open_gate = None
        total_gate_act = Variable(torch.zeros(batch_size,1))
        # output = Variable(torch.zeros(batch_size,self.n_output_neurons))
        output = Variable(torch.FloatTensor(np.full((batch_size, self.n_output_neurons), None)))
        if b_use_cuda:
            total_gate_act = total_gate_act.cuda()
            output = output.cuda()

        if return_gate_status:
            gate_status = np.full((batch_size,) + self.bank_conn.shape, False)

        # Batch norm the inputs.
        if b_batch_norm:
            x = self.input_batch_norm(x)

        # Update activations of all the input banks. These are not gated. Do not apply
        # activation non-linearity here, as that is done in the loop below.
        for i_input_bank in self.idx_input_banks:
            bank_data_acts[i_input_bank] = self.input2hidden[i_input_bank](x)

        # # Update activations of all the input banks. These are not gated.
        # for i_input_bank in self.idx_input_banks:
        #     module_name = 'input_b%0.2d_data' % (i_input_bank)
        #     bank_data_acts[i_input_bank] = F.relu(getattr(self, module_name)(x))

        # Update activations of all the hidden banks. These are gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Check to see if all source bank activations are None, in which case
            # nothing has to be done.
            ### TODO BUG!?:  For unknown reason, the line below is resulting in errors
            # in the computation of the gate activations in the loop below. WTF?
            # Code will work without it, however. It will just be minimally slower
            # because it will loop over all the source banks, checking to see if
            # they are None.
            # if np.all(bank_data_acts[idx_source]==None):
            #     continue

            # Compute gate values for each of the source banks, and data values if
            # gate is open.
            for i_source in idx_source:
                if bank_data_acts[i_source] is not None:
                    dropout_act = self.hidden2hidden_gate_dropout[i_source][i_target](bank_data_acts[i_source])
                    gate_act = self.hidden2hidden_gate[i_source][i_target](dropout_act)

                    ## Apply hard sigmoid or RELU
                    # gate_act = F.relu(gate_act)
                    gate_act = F.hardtanh(gate_act, 0.0, 1.0)
                    if b_neg_gate_loss:
                        total_gate_act -= gate_act
                    else:
                        total_gate_act += gate_act

                    if return_gate_status:
                        # Gate status is set to True (open) only if both the gate node is
                        # greater than zero and one or more activations from the source
                        # bank are non-zero.  I.e., if the gate is open but all the data
                        # inputs are zeros, this is functionally the same as a closed gate.
                        # gate_status[:, i_source, i_target] = (gate_act.data.cpu().numpy()[:,0] > 0) & \
                        #                                      np.any(bank_data_acts[i_source].data, axis=1)
                        gate_status[:, i_source, i_target] = (gate_act.data.cpu().numpy()[:,0] > 0)
                        # if not gate_status[:, i_source, i_target]:
                        #     pdb.set_trace()


                    # Compute data if gate is open
                    if gate_act.data[0,0] > 0:
                        n_open_gates += 1

                        dropout_act = self.hidden2hidden_data_dropout[i_source][i_target](bank_data_acts[i_source])
                        data_act = self.hidden2hidden_data[i_source][i_target](dropout_act)

                        if bank_data_acts[i_target] is None:
                            bank_data_acts[i_target] = gate_act * data_act
                        else:
                            bank_data_acts[i_target] += gate_act * data_act

            if bank_data_acts[i_target] is not None:
                bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
                if not np.any(bank_data_acts[i_target].data):
                    bank_data_acts[i_target] = None
                elif b_batch_norm:
                    bank_data_acts[i_target] = self.hidden_batch_scale[i_target](bank_data_acts[i_target])

        prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size)

        # Update activations of the output layer. The output banks are not gated.
        for i_output_neuron, i_output_bank in enumerate(self.idx_output_banks):
            if bank_data_acts[i_output_bank] is not None:
                data_act = self.hidden2output[i_output_bank](bank_data_acts[i_output_bank])
                output[:,i_output_neuron] = data_act

        # Should we gate the one-to-one outputs?  Just trying RELU for now...
        output = F.relu(output)

        total_gate_act /= self.n_bank_conn  # average per connection

        if return_gate_status:
            return output, total_gate_act, prob_open_gate, gate_status
        else:
            return output, total_gate_act, prob_open_gate

    def save_model(self, model_base_filename):
        # Just saving the model, not the optimizer state. To stop and 
        # resume training, optimizer state needs to be saved as well.
        # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610
        param_dict = {
            'n_input_neurons':self.n_input_neurons,
            'idx_input_banks':self.idx_input_banks,
            'bank_conn':self.bank_conn,
            'idx_output_banks':self.idx_output_banks,
            'n_neurons_per_hidd_bank':self.n_neurons_per_hidd_bank
        }
        torch.save(self.state_dict(), '%s.tch' % (model_base_filename))
        np.save('%s.npy' % (model_base_filename), param_dict)

    # def bias_limit(self, a_min, a_max):
    #     # Limit biases of hidden banks
    #     for i_target in range(self.n_hidd_banks):
    #         idx_source = np.where(self.bank_conn[:,i_target])[0]
    #         for i_source in idx_source:
    #             if self.hidden2hidden_data[i_source][i_target].bias is not None:
    #                 self.hidden2hidden_data[i_source][i_target].bias.data = \
    #                     np.clip(self.hidden2hidden_data[i_source][i_target].bias.data, a_min, a_max)
    #     # Limit biases of output banks
    #     for i_output_bank in self.idx_output_banks:
    #         if self.hidden2output[i_output_bank].bias is not None:
    #             self.hidden2output[i_output_bank].bias.data = \
    #                 np.clip(self.hidden2output[i_output_bank].bias.data, a_min, a_max)


class RouteNetRecurrentGate(nn.Module):
    def __init__(self, n_input_neurons, idx_input_banks, bank_conn, 
                 idx_output_banks, n_output_neurons, n_neurons_per_hidd_bank=10):
        super(RouteNetRecurrentGate, self).__init__()

        self.n_input_neurons = n_input_neurons
        self.idx_input_banks = idx_input_banks
        self.bank_conn = bank_conn
        self.idx_output_banks = idx_output_banks
        self.n_output_neurons = n_output_neurons
        self.n_neurons_per_hidd_bank = n_neurons_per_hidd_bank

        # "bank_conn" defines the connectivity of the banks. This is an NxN boolean matrix for 
        # which a True value in the i,j-th entry indictes that bank i is a source of input to
        # bank j. The matrix could define any structure of banks, including for example, a
        # feedforward layered structure or a structure in which all banks are connected.
        n_hidd_banks = bank_conn.shape[0]
        assert (len(bank_conn.shape) == 2), "bank_conn connectivity matrix must have two dimensions of equal size."
        assert (bank_conn.shape[1] == n_hidd_banks), "bank_conn connectivity matrix must have two dimensions of equal size."

        self.n_hidd_banks = n_hidd_banks
        self.n_bank_conn = np.sum(bank_conn)
        self.prob_dropout_data = 0.0
        self.prob_dropout_gate = 0.0

        # Create all the hidden nn.Linear modules including those for data and those for gates.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        # Use dropout?  Apply same single dropout to each source?  Each source/target combo?
        # Each source/target combo and each source/gate combo?
        self.hidden2hidden_gate = nn.ModuleList()
        self.hidden2hidden_data = nn.ModuleList()
        self.hidden2hidden_gate_dropout = nn.ModuleList()
        self.hidden2hidden_data_dropout = nn.ModuleList()
        self.hidden_batch_norm = nn.ModuleList()
        for i_source in range(n_hidd_banks):
            self.hidden2hidden_gate.append(nn.ModuleList())
            self.hidden2hidden_data.append(nn.ModuleList())
            self.hidden2hidden_gate_dropout.append(nn.ModuleList())
            self.hidden2hidden_data_dropout.append(nn.ModuleList())
            self.hidden_batch_norm.append(nn.BatchNorm1d(self.n_neurons_per_hidd_bank))
            for i_target in range(n_hidd_banks):
                if bank_conn[i_source, i_target]:
                    self.hidden2hidden_gate[i_source].append(nn.Linear(n_neurons_per_hidd_bank, 1, bias=False))
                    self.hidden2hidden_data[i_source].append(nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank, bias=True))
                    self.hidden2hidden_gate_dropout[i_source].append(nn.Dropout(p=self.prob_dropout_gate))
                    self.hidden2hidden_data_dropout[i_source].append(nn.Dropout(p=self.prob_dropout_data))
                else:
                    self.hidden2hidden_gate[i_source].append(None)
                    self.hidden2hidden_data[i_source].append(None)
                    self.hidden2hidden_gate_dropout[i_source].append(None)
                    self.hidden2hidden_data_dropout[i_source].append(None)

        # Create the connections between inputs and banks that receive inputs
        self.input_batch_norm = nn.BatchNorm1d(self.n_input_neurons)
        self.input2hidden = nn.ModuleList()
        for i_input_bank in range(n_hidd_banks):
            self.input2hidden.append(None)
        for i_input_bank in idx_input_banks:
            # TODO: Should layers between inputs and receiving banks have a bias or not?
            self.input2hidden[i_input_bank] = nn.Linear(n_input_neurons, n_neurons_per_hidd_bank)

        # Create the connections between output banks and network output layer.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        self.hidden2output = nn.ModuleList()
        for i_output_bank in range(n_hidd_banks):
            self.hidden2output.append(None)
        for i_output_bank in idx_output_banks:
            self.hidden2output[i_output_bank] = nn.Linear(n_neurons_per_hidd_bank, n_output_neurons, bias=False)

    @classmethod
    def init_from_files(cls, model_base_filename):
        # Load model metaparameters, instantiate a model with that architecture,
        # load model weights, and set the model weights.
        param_dict = np.load('%s.npy' % (model_base_filename)).item()
        net = cls(**param_dict)
        # if b_use_cuda:
        #     net = net.cuda()
        net.load_state_dict(torch.load('%s.tch' % (model_base_filename)))
        return net

    def forward_ff_softgate(self, x, return_gate_status=False, b_use_cuda=False):
        # Unlike the main forward() method, this one uses soft gates thus
        # allowing batches to be used in training. The notion is that this
        # could be used for fast pre-training, and then forward() used for
        # final training with hard gating.
        b_batch_norm = True

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension

        bank_data_acts = np.full(self.n_hidd_banks, None)
        n_open_gates = 0
        output = None
        total_gate_act = 0

        if return_gate_status:
            gate_status = np.full((batch_size,) + self.bank_conn.shape, False)

        # Batch norm the inputs.
        if b_batch_norm:
            x = self.input_batch_norm(x)

        # Update activations of all the input banks. These are not gated.
        for i_input_bank in self.idx_input_banks:
            bank_data_acts[i_input_bank] = F.relu(self.input2hidden[i_input_bank](x))

        # Update activations of all the hidden banks. These are soft gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Compute gate values for each of the input banks, and multiply
            # by the incoming activations.
            for idx, i_source in enumerate(idx_source):
                dropout_act = self.hidden2hidden_gate_dropout[i_source][i_target](bank_data_acts[i_source])
                gate_act = self.hidden2hidden_gate[i_source][i_target](dropout_act)
                
                ## Apply hard sigmoid or RELU
                # gate_act = F.relu(gate_act)
                gate_act = F.hardtanh(gate_act, 0.0, 1.0)

                total_gate_act += gate_act

                if return_gate_status:
                    gate_status[:, i_source, i_target] = gate_act.data.cpu().numpy()[:,0] > 0
                    z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
                    n_open_gates += np.sum(z)

                dropout_act = self.hidden2hidden_data_dropout[i_source][i_target](bank_data_acts[i_source])
                data_act = self.hidden2hidden_data[i_source][i_target](dropout_act)

                if bank_data_acts[i_target] is None:
                    bank_data_acts[i_target] = gate_act * data_act
                else:
                    bank_data_acts[i_target] += gate_act * data_act

            bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
            if b_batch_norm:
                bank_data_acts[i_target] = self.hidden_batch_norm[i_target](bank_data_acts[i_target])

        if return_gate_status:
            prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size)

        # Update activations of the output layer. The output banks are not gated.
        for i_output_bank in self.idx_output_banks:
            data_act = self.hidden2output[i_output_bank](bank_data_acts[i_output_bank])

            if output is None:
                output = data_act
            else:
                output += data_act

        total_gate_act /= self.n_bank_conn  # average per connection

        if return_gate_status:
            return output, total_gate_act, prob_open_gate, gate_status
        else:
            return output, total_gate_act

    def forward_fb_softgate(self, x, n_hidden_iters=4, return_gate_status=False, b_use_cuda=False):
        # Unlike the main forward() method, this one uses soft gates thus
        # allowing batches to be used in training. The notion is that this
        # could be used for fast pre-training, and then forward() used for
        # final training with hard gating.
        #
        # This function computes activations for the input banks,
        # then randomly updates activations for hidden banks and gates. This
        # might improve training (regularization) and require fewer training
        # batches versus if the full network is iterated over multiple times.
        # Also, it's not clear what is the best update order to apply
        # to the banks and gates in the network, generally.
        #
        # Another option would be to update all the FF data banks, then
        # randomly update gates and banks.

        b_batch_norm = True

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension

        bank_data_acts = np.full(self.n_hidd_banks, None)
        bank_gate_acts = np.full((self.n_hidd_banks,self.n_hidd_banks), None)
        for i in range(self.n_hidd_banks):
            bank_data_acts[i] = Variable(torch.zeros((batch_size, self.n_output_neurons)))
            if b_use_cuda:
                bank_data_acts[i] = bank_data_acts[i].cuda()
            for j in range(self.n_hidd_banks):
                bank_gate_acts[i,j] = Variable(torch.ones((batch_size, 1))) # initialize with gates open
                if b_use_cuda:
                    bank_gate_acts[i,j] = bank_gate_acts[i,j].cuda()

        n_open_gates = 0
        output = None
        total_gate_act = Variable(torch.zeros(batch_size,1))
        if b_use_cuda:
            total_gate_act = total_gate_act.cuda()

        if return_gate_status:
            gate_status = np.full((batch_size,) + self.bank_conn.shape, False)

        # Batch norm the inputs.
        if b_batch_norm:
            x = self.input_batch_norm(x)

        # Update activations of all the input banks. These are not gated.
        for i_input_bank in self.idx_input_banks:
            bank_data_acts[i_input_bank] = F.relu(self.input2hidden[i_input_bank](x))

        idx_conn_targ = np.unique(np.where(self.bank_conn)[1])
        n_updates = n_hidden_iters * self.n_bank_conn
        i_rand_targets = np.asarray([]).astype(np.int)
        for i in range(n_hidden_iters):
            i_rand_targets = np.append(i_rand_targets, np.random.permutation(idx_conn_targ))
            # i_rand_targets = np.append(i_rand_targets, idx_conn_targ)

        for i_target in i_rand_targets:
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Multiply source bank activations by gate values, and compute new target bank activations
            for idx, i_source in enumerate(idx_source):
                dropout_act = self.hidden2hidden_data_dropout[i_source][i_target](bank_data_acts[i_source])
                data_act = self.hidden2hidden_data[i_source][i_target](dropout_act)
                bank_data_acts[i_target] += bank_gate_acts[i_source,i_target] * data_act

            bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
            if b_batch_norm:
                bank_data_acts[i_target] = self.hidden_batch_norm[i_target](bank_data_acts[i_target])

            # Update gate activations, using updated target activation
            for idx, i_source in enumerate(idx_source):
                dropout_act = self.hidden2hidden_gate_dropout[i_source][i_target](bank_data_acts[i_target])
                gate_act = self.hidden2hidden_gate[i_source][i_target](dropout_act)

                ## Apply hard sigmoid or RELU
                # gate_act = F.relu(gate_act)
                gate_act = F.hardtanh(gate_act, 0.0, 1.0)

                total_gate_act += gate_act

                # TODO: Need to compute this differently, now that the network has loops
                if return_gate_status:
                    gate_status[:, i_source, i_target] = gate_act.data.cpu().numpy()[:,0] > 0
                    z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
                    n_open_gates += np.sum(z)

        if return_gate_status:
            prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size * n_hidden_iters)

        # Update activations of the output layer. The output banks are not gated.
        for i_output_bank in self.idx_output_banks:
            data_act = self.hidden2output[i_output_bank](bank_data_acts[i_output_bank])

            if output is None:
                output = data_act
            else:
                output += data_act

        total_gate_act /= self.n_bank_conn  # average per connection

        if return_gate_status:
            return output, total_gate_act, prob_open_gate, gate_status
        else:
            return output, total_gate_act

    def forward_recurrent_softgate(self, x, n_hidden_iters=4, return_gate_status=False, b_use_cuda=False):
        # Unlike the main forward() method, this one uses soft gates thus
        # allowing batches to be used in training. The notion is that this
        # could be used for fast pre-training, and then forward() used for
        # final training with hard gating.
        #
        # This functions has gates that take input from both the source and
        # data banks. Data banks are initialized as zero, so in the first
        # iteration of updates, only the FF path impacts the gating (sort
        # of, depending on activation update ordering: FF or random).
        #
        # This function randomly updates activations for hidden banks and gates.
        # This might improve training (regularization) and require fewer training
        # batches versus if the full network is iterated over multiple times.
        # Also, it's not clear what is the best update order to apply
        # to the banks and gates in the network, generally.
        #
        # Another option would be to update all the FF data banks, then
        # randomly update gates and banks.

        b_batch_norm = True

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension

        ##############################################################################
        ##############################################################################
        # TODO: Activation gates based on both FF and FB data banks. Don't need
        # to store gate activations, as in fb_softgate()? Or could store both
        # ff and fb components before nonlinear activation function. When a bank
        # is updated we: (1) Update FB gate components, (2) add stored FF gate components
        # and apply nonlinearity, (3) multiply source bank activation by gates,
        # (4) get updated target bank activation, (5) use new target bank activations
        # to update it's downtream FF gate components.
        ##############################################################################
        ##############################################################################

        bank_data_acts = np.full(self.n_hidd_banks, None)
        # bank_gate_acts = np.full((self.n_hidd_banks,self.n_hidd_banks), None)
        for i in range(self.n_hidd_banks):
            bank_data_acts[i] = Variable(torch.zeros((batch_size, self.n_output_neurons)))
            if b_use_cuda:
                bank_data_acts[i] = bank_data_acts[i].cuda()
            # for j in range(self.n_hidd_banks):
            #     bank_gate_acts[i,j] = Variable(torch.ones((batch_size, 1))) # initialize with gates open
            #     if b_use_cuda:
            #         bank_gate_acts[i,j] = bank_gate_acts[i,j].cuda()

        n_open_gates = 0
        output = None
        total_gate_act = Variable(torch.zeros(batch_size,1))
        if b_use_cuda:
            total_gate_act = total_gate_act.cuda()

        if return_gate_status:
            gate_status = np.full((batch_size,) + self.bank_conn.shape, False)

        # Batch norm the inputs.
        if b_batch_norm:
            x = self.input_batch_norm(x)

        # Update activations of all the input banks. These are not gated.
        for i_input_bank in self.idx_input_banks:
            bank_data_acts[i_input_bank] = F.relu(self.input2hidden[i_input_bank](x))

        idx_conn_targ = np.unique(np.where(self.bank_conn)[1])
        n_updates = n_hidden_iters * self.n_bank_conn
        i_rand_targets = np.asarray([]).astype(np.int)
        for i in range(n_hidden_iters):
            i_rand_targets = np.append(i_rand_targets, np.random.permutation(idx_conn_targ))
            # i_rand_targets = np.append(i_rand_targets, idx_conn_targ)

        for i_target in i_rand_targets:
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Multiply source bank activations by gate values, and compute new target bank activations
            for idx, i_source in enumerate(idx_source):
                dropout_act = self.hidden2hidden_data_dropout[i_source][i_target](bank_data_acts[i_source])
                data_act = self.hidden2hidden_data[i_source][i_target](dropout_act)
                bank_data_acts[i_target] += bank_gate_acts[i_source,i_target] * data_act

            bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
            if b_batch_norm:
                bank_data_acts[i_target] = self.hidden_batch_norm[i_target](bank_data_acts[i_target])

            # Update gate activations, using updated target activation
            for idx, i_source in enumerate(idx_source):
                dropout_act = self.hidden2hidden_gate_dropout[i_source][i_target](bank_data_acts[i_target])
                gate_act = self.hidden2hidden_gate[i_source][i_target](dropout_act)

                ## Apply hard sigmoid or RELU
                # gate_act = F.relu(gate_act)
                gate_act = F.hardtanh(gate_act, 0.0, 1.0)

                total_gate_act += gate_act

                # TODO: Need to compute this differently, now that the network has loops
                if return_gate_status:
                    gate_status[:, i_source, i_target] = gate_act.data.cpu().numpy()[:,0] > 0
                    z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
                    n_open_gates += np.sum(z)

        if return_gate_status:
            prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size * n_hidden_iters)

        # Update activations of the output layer. The output banks are not gated.
        for i_output_bank in self.idx_output_banks:
            data_act = self.hidden2output[i_output_bank](bank_data_acts[i_output_bank])

            if output is None:
                output = data_act
            else:
                output += data_act

        total_gate_act /= self.n_bank_conn  # average per connection

        if return_gate_status:
            return output, total_gate_act, prob_open_gate, gate_status
        else:
            return output, total_gate_act

    def save_model(self, model_base_filename):
        # Just saving the model, not the optimizer state. To stop and 
        # resume training, optimizer state needs to be saved as well.
        # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610
        param_dict = {
            'n_input_neurons':self.n_input_neurons,
            'idx_input_banks':self.idx_input_banks,
            'bank_conn':self.bank_conn,
            'idx_output_banks':self.idx_output_banks,
            'n_output_neurons':self.n_output_neurons,
            'n_neurons_per_hidd_bank':self.n_neurons_per_hidd_bank
        }
        torch.save(self.state_dict(), '%s.tch' % (model_base_filename))
        np.save('%s.npy' % (model_base_filename), param_dict)


class RandomLocationMNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, expanded_size=56, group_size_1D=56/4, xy_resolution=10, rotate=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.expanded_size = expanded_size
        self.xy_resolution = xy_resolution
        self.pad_each_side = expanded_size - 28
        self.group_size_per_side = group_size_1D
        self.groups_per_side = expanded_size/group_size_1D
        self.rotate = rotate

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.rotate:
            img = img.transpose_(1,0)
            
        # MJR:
        # Padding and random cropping the image here, rather than with a transform.
        # After cropping, image size will be 28+pad_each_side, on each side.
        img = np.pad(img.numpy(), self.pad_each_side, 'constant', constant_values=0)
        # left = random.randint(0, self.pad_each_side-1)
        # top = random.randint(0, self.pad_each_side-1)
        left = random.randint(0, self.pad_each_side)
        top = random.randint(0, self.pad_each_side)
        img = img[top:top+self.expanded_size, left:left+self.expanded_size]
        img = torch.ByteTensor(img)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        # MJR:
        # Reshape img into stack of tiles/groups, where the tiles are taken
        # from the larger image first by rows, then by columns(e.g., top-left
        # tile, ..., bottom-left tile, ..., top-right tile, ...,
        # bottom-right tile). Tiles do not overlap.
        img_grouped = img.view((1, self.groups_per_side, self.group_size_per_side, self.groups_per_side, self.group_size_per_side))
        img_grouped = img_grouped.permute(0, 1, 3, 2, 4).contiguous()
        img_grouped = img_grouped.view((1, self.groups_per_side**2, self.group_size_per_side, self.group_size_per_side)).contiguous()
        # And repackage into a tuple of tiles. Must be a better way to do this...
        img = ()
        for i_group in range(self.groups_per_side**2):
            img += (img_grouped[:,i_group,:,:],)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # MJR: Target is array of (label, idx_center_x, idx_center_y)
        idx_center_x = self.expanded_size-left-14
        idx_center_y = self.expanded_size-top-14
        idx_center_x = int(round(idx_center_x / float(self.expanded_size-1.0) * (self.xy_resolution-1.0)))
        idx_center_y = int(round(idx_center_y / float(self.expanded_size-1.0) * (self.xy_resolution-1.0)))
        target = np.asarray((target, idx_center_x, idx_center_y))

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')
