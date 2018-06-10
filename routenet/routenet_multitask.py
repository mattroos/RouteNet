# routnet_multitask.py

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


def earth_mover_loss(prediction, target, b_use_cuda=False):
    # Assumes that target is a scalar, indicating which node
    # should have all the probability mass (target distribution
    # as 1 at one node, 0 at all others).
    n_labels = prediction.size()[1]
    locations = Variable(torch.arange(0,n_labels).view(1,-1)).float()
    if b_use_cuda:
        locations = locations.cuda()
    dist_targ_to_loc = torch.abs(target.view(-1,1).float() - locations)
    pmf = F.softmax(prediction, dim=1)
    loss_dist = torch.mean(torch.sum(pmf * dist_targ_to_loc, dim=1)) / n_labels
    return loss_dist

def earth_mover_loss2(prediction, target, b_use_cuda=False):
    # Assumes that target is a scalar, indicating which node
    # should have all the probability mass (target distribution
    # as 1 at one node, 0 at all others).
    #
    # Formulate calculation such that target distribution 
    # doesn't have to be one-hot.
    batch_size = prediction.size()[0]
    n_labels = prediction.size()[1]
    pmf = F.softmax(prediction, dim=1)

    pmf_targ = np.zeros((batch_size, n_labels), dtype=np.float32)
    pmf_targ[np.arange(batch_size), target.data.cpu().numpy().astype(np.int)] = 1
    pmf_targ = Variable(torch.from_numpy(pmf_targ))
    if b_use_cuda:
        pmf_targ = pmf_targ.cuda()

    d = pmf - pmf_targ
    dd = torch.cumsum(d, dim=1)
    loss_dist_sample = torch.sum(torch.abs(dd), dim=1)
    loss_dist = torch.mean(loss_dist_sample) / n_labels
    return loss_dist

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
    
    # Build connectivity sub-matrix and insert it into the full
    # connectivity matrix for each pair of connected layers.
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

###################################################
# TODO:
# Add batch norm to front of banks
# Create new class that uses ModuleList?
###################################################

class RouteNet(nn.Module):
    def __init__(self, n_input_neurons, idx_input_banks, bank_conn, 
                 idx_output_banks, n_output_neurons, n_neurons_per_hidd_bank=10):
        super(RouteNet, self).__init__()
        # Allowing multiple targets (that is, different groups of output neurons
        # for different tasks), so idx_output_neurons is list of arrays and 
        # n_output_neurons is a list of equal length.

        # If just a single task is requested, converted some inputs to lists
        # anyway for compatibility with multi-task code.
        if not isinstance(n_output_neurons, list):
            n_output_neurons = [n_output_neurons]
            idx_output_banks = [idx_output_banks]
            self.n_tasks = 1
        else:
            self.n_tasks = len(n_output_neurons)

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
        for i_source in range(n_hidd_banks):
            for i_target in range(n_hidd_banks):
                if bank_conn[i_source, i_target]:
                    module_name = 'b%0.2d_b%0.2d_gate_dropout' % (i_source, i_target)
                    setattr(self, module_name, nn.Dropout(p=self.prob_dropout_gate))

                    # Use unbiased gates, so hard gating is equivalent to soft gating
                    # such that for hard gating, if all input gates to a target bank
                    # are closed, then that bank be inactive and the gates downstream
                    # from the target bank will also be closed.
                    module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
                    setattr(self, module_name, nn.Linear(n_neurons_per_hidd_bank, 1, bias=False))

                    module_name = 'b%0.2d_b%0.2d_data_dropout' % (i_source, i_target)
                    setattr(self, module_name, nn.Dropout(p=self.prob_dropout_data))

                    module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
                    setattr(self, module_name, nn.Linear(n_neurons_per_hidd_bank, n_neurons_per_hidd_bank, bias=True))

        # Create the connections between inputs and banks that receive inputs
        for i_input_bank in idx_input_banks:
            module_name = 'input_b%0.2d_data' % (i_input_bank)
            # TODO: Should layers between inputs and receiving banks have a bias or not?
            setattr(self, module_name, nn.Linear(n_input_neurons, n_neurons_per_hidd_bank))

        # Create the connections between output banks and network output layer.
        # Do not use a bias, so hard gating will be equivalent to soft gating.
        for i_task, idx in enumerate(idx_output_banks):
            for i_output_bank in idx:
                module_name = 'b%0.2d_t%0.2d_output_data' % (i_output_bank, i_task)
                setattr(self, module_name, nn.Linear(n_neurons_per_hidd_bank, n_output_neurons[i_task], bias=False))

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

    def forward_hardgate(self, x, return_gate_status=False):
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

        bank_data_acts = np.full(self.n_hidd_banks, None)
        n_open_gates = 0
        total_gate_act = None

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension
        
        if return_gate_status:
            gate_status = np.full(self.bank_conn.shape, False)

        # Update activations of all the input banks. These are not gated.
        for i_input_bank in self.idx_input_banks:
            module_name = 'input_b%0.2d_data' % (i_input_bank)
            bank_data_acts[i_input_bank] = F.relu(getattr(self, module_name)(x))

        # Update activations of all the hidden banks. These are gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Check to see if all source bank activations are None, in which case
            # nothing has to be done.
            if np.all(bank_data_acts[idx_source]==None):
                continue

            # Compute gate values for each of the input banks, and data values if
            # gate is open.
            for i_source in idx_source:
                if bank_data_acts[i_source] is not None:
                    # module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
                    # gate_act = getattr(self, module_name)(bank_data_acts[i_source])
                    module_name = 'b%0.2d_b%0.2d_gate_dropout' % (i_source, i_target)
                    dropout_act = getattr(self, module_name)(bank_data_acts[i_source])
                    module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
                    gate_act = getattr(self, module_name)(dropout_act)

                    ## Apply hard sigmoid, RELU, or similar
                    # gate_act = F.relu(gate_act)
                    # gate_act = F.sigmoid(gate_act)
                    gate_act = F.hardtanh(gate_act, 0.0, 1.0)

                    if total_gate_act is None:
                        total_gate_act = gate_act
                    else:
                        total_gate_act += gate_act

                    if gate_act.data[0,0] > 0:
                        if return_gate_status:
                            gate_status[i_source, i_target] = True
                            n_open_gates += 1
                        # module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
                        # data_act = getattr(self, module_name)(bank_data_acts[i_source])
                        module_name = 'b%0.2d_b%0.2d_data_dropout' % (i_source, i_target)
                        dropout_act = getattr(self, module_name)(bank_data_acts[i_source])
                        module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
                        data_act = getattr(self, module_name)(dropout_act)

                        # Could we train in a way such that gate activations
                        # are nearly always 0 or 1, so we don't have to multiply
                        # them by the data activations, as in the soft-gating
                        # case? Probabilistic activation?
                        if bank_data_acts[i_target] is None:
                            # TODO: If gate_act==1.0, don't need to multiply data activations by gate activation
                            # Some bug below. Can't multiply by gate_act and also do backprop (loss.backward()).
                            #   Don't know why.
                            bank_data_acts[i_target] = gate_act * data_act
                            # bank_data_acts[i_target] = data_act
                        else:
                            # TODO: If gate_act==1.0, don't need to multiply data activations by gate activation
                            bank_data_acts[i_target] += gate_act * data_act

            if bank_data_acts[i_target] is not None:
                bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])

        if return_gate_status:
            prob_open_gate = n_open_gates / float(self.n_bank_conn)

        # Update activations output layer. The output 'bank' is not gated, but it may
        # not have any active inputs, so have to check for that.
        output = None
        if np.all(bank_data_acts[self.idx_output_banks]==None):
            return output, total_gate_act, prob_open_gate, gate_status
        for i_output_bank in self.idx_output_banks:
            if bank_data_acts[i_output_bank] is not None:
                module_name = 'b%0.2d_output_data' % (i_output_bank)
                data_act = getattr(self, module_name)(bank_data_acts[i_output_bank])
                if output is None:
                    output = data_act
                else:
                    output += data_act

        if return_gate_status:
            return output, total_gate_act, prob_open_gate, gate_status
        else:
            return output, total_gate_act

    # def forward_softgate(self, x):
    #     # Unlike the main forward() method, this one uses soft gates thus
    #     # allowing batches to be used in training. The notion is that this
    #     # could be used for fast pre-training, and then forward() used for
    #     # final training with hard gating.

    #     bank_data_acts = np.full(self.n_hidd_banks, None)
    #     n_open_gates = 0
    #     output = None
    #     total_gate_act = 0

        # batch_size = x.size()[0]
        # x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension

    #     gate_status = np.full((batch_size,) + self.bank_conn.shape, False)

    #     # Update activations of all the input banks. These are not gated.
    #     for i_input_bank in self.idx_input_banks:
    #         module_name = 'input_b%0.2d_data' % (i_input_bank)
    #         bank_data_acts[i_input_bank] = F.relu(getattr(self, module_name)(x))
    #         # TODO: Add activations to total activation energy?

    #     # Update activations of all the hidden banks. These are soft gated.
    #     for i_target in range(self.n_hidd_banks):
    #         # Get list of source banks that are connected to this target bank
    #         idx_source = np.where(self.bank_conn[:,i_target])[0]

    #         # Compute gate values for each of the input banks, and multiply
    #         # by the incoming activations.
    #         for i_source in idx_source:
    #             # module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
    #             # gate_act = getattr(self, module_name)(bank_data_acts[i_source])
    #             module_name = 'b%0.2d_b%0.2d_gate_dropout' % (i_source, i_target)
    #             dropout_act = getattr(self, module_name)(bank_data_acts[i_source])
    #             module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
    #             gate_act = getattr(self, module_name)(dropout_act)
                
    #             ## Apply hard sigmoid, RELU, or similar
    #             # gate_act = F.relu(gate_act)
    #             # gate_act = F.sigmoid(gate_act)
    #             gate_act = F.hardtanh(gate_act, 0.0, 1.0)
    #             total_gate_act += gate_act
    #             # TODO: Add activations to total activation energy?

    #             gate_status[:, i_source, i_target] = gate_act.data.cpu().numpy()[:,0] > 0

    #             z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
    #             n_open_gates += np.sum(z)

    #             # module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
    #             # data_act = getattr(self, module_name)(bank_data_acts[i_source])
    #             module_name = 'b%0.2d_b%0.2d_data_dropout' % (i_source, i_target)
    #             dropout_act = getattr(self, module_name)(bank_data_acts[i_source])
    #             module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
    #             data_act = getattr(self, module_name)(dropout_act)

    #             if bank_data_acts[i_target] is None:
    #                 bank_data_acts[i_target] = gate_act * data_act
    #             else:
    #                 bank_data_acts[i_target] += gate_act * data_act

    #         bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
    #         # TODO: Add activations to total activation energy?

    #     prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size)

    #     # Update activations of the output layer. The output banks are not gated.
    #     for i_output_bank in self.idx_output_banks:
    #         module_name = 'b%0.2d_output_data' % (i_output_bank)
    #         # Part of BUG is here/below. If output layers have bias, then even if
    #         # input is zero, they may have non-zero output, unlike for the
    #         # hard gate model.
    #         data_act = getattr(self, module_name)(bank_data_acts[i_output_bank])
    #         if output is None:
    #             output = data_act
    #         else:
    #             output += data_act

    #     # TODO: Add output activations to total activation energy?

    #     return output, total_gate_act, prob_open_gate, gate_status

    def forward_softgate(self, x, return_gate_status=False):
        # Unlike the main forward() method, this one uses soft gates thus
        # allowing batches to be used in training. The notion is that this
        # could be used for fast pre-training, and then forward() used for
        # final training with hard gating.

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)  # Flatten across all dimensions except batch dimension

        bank_data_acts = np.full(self.n_hidd_banks, None)
        n_open_gates = 0
        output = [None] * self.n_tasks
        total_gate_act = 0

        if return_gate_status:
            gate_status = np.full((batch_size,) + self.bank_conn.shape, False)

        # Update activations of all the input banks. These are not gated.
        for i_input_bank in self.idx_input_banks:
            module_name = 'input_b%0.2d_data' % (i_input_bank)
            bank_data_acts[i_input_bank] = F.relu(getattr(self, module_name)(x))
            # bank_acts_name = 'b%0.2d_acts' % (i_input_bank)
            # setattr(self, bank_acts_name, F.relu(getattr(self, module_name)(x)))

        # Update activations of all the hidden banks. These are soft gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            target_bank_acts_name = 'b%0.2d_acts' % (i_target)

            # Compute gate values for each of the input banks, and multiply
            # by the incoming activations.
            for idx, i_source in enumerate(idx_source):
                # module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
                # gate_act = getattr(self, module_name)(bank_data_acts[i_source])
                module_name = 'b%0.2d_b%0.2d_gate_dropout' % (i_source, i_target)
                dropout_act = getattr(self, module_name)(bank_data_acts[i_source])
                # source_bank_acts_name = 'b%0.2d_acts' % (i_source)
                # dropout_act = getattr(self, module_name)(getattr(self, source_bank_acts_name))
                module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
                gate_act = getattr(self, module_name)(dropout_act)
                
                ## Apply hard sigmoid or RELU
                # gate_act = F.relu(gate_act)
                gate_act = F.hardtanh(gate_act, 0.0, 1.0)

                total_gate_act += gate_act

                if return_gate_status:
                    gate_status[:, i_source, i_target] = gate_act.data.cpu().numpy()[:,0] > 0
                    z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
                    n_open_gates += np.sum(z)

                # module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
                # data_act = getattr(self, module_name)(bank_data_acts[i_source])
                module_name = 'b%0.2d_b%0.2d_data_dropout' % (i_source, i_target)
                dropout_act = getattr(self, module_name)(bank_data_acts[i_source])
                # dropout_act = getattr(self, module_name)(getattr(self, source_bank_acts_name))
                module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
                data_act = getattr(self, module_name)(dropout_act)

                if bank_data_acts[i_target] is None:
                    bank_data_acts[i_target] = gate_act * data_act
                else:
                    bank_data_acts[i_target] += gate_act * data_act
                # if idx==0:
                #     setattr(self, target_bank_acts_name, gate_act * data_act)
                # else:
                #     setattr(self, target_bank_acts_name, getattr(self, target_bank_acts_name) + gate_act * data_act)

            bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
            # setattr(self, target_bank_acts_name, F.relu(getattr(self, target_bank_acts_name)))


        if return_gate_status:
            prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size)

        # Update activations of the output layer. The output banks are not gated.
        for i_task, idx in enumerate(self.idx_output_banks):
            for i_output_bank in idx:
                module_name = 'b%0.2d_t%0.2d_output_data' % (i_output_bank, i_task)
                data_act = getattr(self, module_name)(bank_data_acts[i_output_bank])
                # bank_acts_name = 'b%0.2d_acts' % (i_output_bank)
                # data_act = getattr(self, module_name)(getattr(self, bank_acts_name))

                if output[i_task] is None:
                    output[i_task] = data_act
                else:
                    output[i_task] += data_act


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

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, expanded_size=56, xy_resolution=10):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.expanded_size = expanded_size
        self.xy_resolution = xy_resolution
        self.pad_each_side = expanded_size - 28

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

        # MJR:
        # Padding and random cropping the image here, rather than with a transform.
        # After cropping, image size will be 28+pad_each_side, on each side.
        img = np.pad(img.numpy(), self.pad_each_side, 'constant', constant_values=0)
        left = random.randint(0, self.pad_each_side-1)
        top = random.randint(0, self.pad_each_side-1)
        img = img[top:top+self.expanded_size, left:left+self.expanded_size]
        img = torch.ByteTensor(img)


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

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
