# routnet.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import sys


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
        self.n_bank_conn = np.sum(bank_conn)
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
            idx_source = np.where(self.bank_conn[:,i_target])[0]

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
            prob_open_gate = n_open_gates / float(self.n_bank_conn)
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


    def forward_softgate(self, x):
        # Unlike the main forward() method, this one uses soft gates thus
        # allowing batches to be used in training. The notion is that this
        # could be used for fast pre-training, and then forward() used for
        # final training with hard gating.

        bank_data_acts = np.full(self.n_hidd_banks, None)
        prob_open_gate = 0
        n_open_gates = 0
        output = None
        total_gate_act = 0

        x = x.view(-1, 784)
        batch_size = len(x)

        # Update activations of all the input banks. These are not gated.
        for i_input_bank in self.idx_input_banks:
            module_name = 'input_b%0.2d_data' % (i_input_bank)
            bank_data_acts[i_input_bank] = F.relu(getattr(self, module_name)(x))
            # TODO: Add activations to total activation energy?

        # Update activations of all the hidden banks. These are soft gated.
        for i_target in range(self.n_hidd_banks):
            # Get list of source banks that are connected to this target bank
            idx_source = np.where(self.bank_conn[:,i_target])[0]

            # Compute gate values for each of the input banks, and multiply
            # by the incoming activations.
            for i_source in idx_source:
                module_name = 'b%0.2d_b%0.2d_gate' % (i_source, i_target)
                gate_act = getattr(self, module_name)(bank_data_acts[i_source])
                
                # Apply hard sigmoid, RELU, or similar
                gate_act = F.relu(gate_act)
                # gate_act = F.sigmoid(gate_act)
                # gate_act = F.hardtanh(gate_act, 0.0, 1.0)
                total_gate_act += gate_act
                # TODO: Add activations to total activation energy?

                z = (gate_act.data.cpu().numpy()>0).flatten().astype(np.int)
                n_open_gates += np.sum(z)

                module_name = 'b%0.2d_b%0.2d_data' % (i_source, i_target)
                data_act = getattr(self, module_name)(bank_data_acts[i_source])

                if bank_data_acts[i_target] is None:
                    bank_data_acts[i_target] = gate_act * data_act
                else:
                    bank_data_acts[i_target] += gate_act * data_act

            bank_data_acts[i_target] = F.relu(bank_data_acts[i_target])
            # TODO: Add activations to total activation energy?

        # Update activations of the output layer. The output banks are not gated.
        for i_output_bank in self.idx_output_banks:
            module_name = 'b%0.2d_output_data' % (i_output_bank)
            data_act = getattr(self, module_name)(bank_data_acts[i_output_bank])
            if output is None:
                output = data_act
            else:
                output += data_act

        output = F.log_softmax(output, dim=1)
        prob_open_gate = n_open_gates / float((self.n_bank_conn) * batch_size)
        # TODO: Add output activations to total activation energy?

        return output, total_gate_act, prob_open_gate

