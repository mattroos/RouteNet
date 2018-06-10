
# temp_localize.py
#
# Predict location of object

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time

plt.ion()

b_use_cuda = torch.cuda.is_available()

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if b_use_cuda:
    torch.cuda.manual_seed(seed)


def make_image_batch(batch_size, field_size=56, obj_size=6):
    assert obj_size%2==0, 'make_image_batch(): obj_size must be even number.'
    im = np.zeros((field_size, field_size))
    im[0:obj_size, 0:obj_size] = 1.
    data = np.zeros((batch_size, field_size, field_size), dtype=np.float32)
    # im = np.zeros((1, field_size))
    # im[0, 0:obj_size] = 1.
    # data = np.zeros((batch_size, 1, field_size), dtype=np.float32)
    targ_x = np.random.randint(obj_size/2, high=field_size-obj_size/2+1, size=batch_size)
    targ_y = np.random.randint(obj_size/2, high=field_size-obj_size/2, size=batch_size)
    for i in range(batch_size):
        data[i,:,:] = np.roll(im, targ_x[i]-obj_size/2, axis=1) # shift along x-axis
        data[i,:,:] = np.roll(data[i,:,:], targ_y[i]-obj_size/2, axis=0) # shift along y-axis
    data = torch.from_numpy(data)
    # targ_x = torch.from_numpy(targ_x.astype(np.float32)) / float(field_size) - 0.5
    # targ_x = 6.0 * targ_x
    targ_x = targ_x - obj_size/2    # targets as integers starting for 0
    targ_x = torch.from_numpy(targ_x.astype(np.int))
    if b_use_cuda:
        data, targ_x = data.cuda(), targ_x.cuda()
    return data, targ_x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

field_size = 56
obj_size = 14
n_labels = field_size-obj_size+1

# TODO: Specify number of labels/outputs and divide the space up
# into chunks that span more than one pixel. Might want have targets
# that are more than one-hot.



## Define earth mover loss.
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
    pmf_targ[np.arange(batch_size), target.data.cpu().numpy()] = 1
    pmf_targ = Variable(torch.from_numpy(pmf_targ))
    if b_use_cuda:
        pmf_targ = pmf_targ.cuda()

    d = pmf - pmf_targ
    dd = torch.cumsum(d, dim=1)
    loss_dist_sample = torch.sum(torch.abs(dd), dim=1)
    loss_dist = torch.mean(loss_dist_sample) / n_labels
    return loss_dist


# Try without nn.Sequential.
class MyNet(nn.Module):
    def __init__(self, n_outputs, field_size=56, n_hidden=10, bias=True, p_dropout=0.5):
        super(MyNet, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(field_size**2)
        self.linear_hid1 = nn.Linear(field_size**2, n_hidden, bias=bias)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=p_dropout)
        self.batch_norm2 = nn.BatchNorm1d(n_hidden)
        self.linear_hid2 = nn.Linear(n_hidden, n_hidden, bias=bias)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=p_dropout)
        self.batch_norm3 = nn.BatchNorm1d(n_hidden)
        self.linear_out = nn.Linear(n_hidden, n_outputs, bias=bias)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        # hidden = self.linear_hid1(x.view(batch_size, -1))
        norm = self.batch_norm1(x.view(batch_size, -1))
        hidden = self.linear_hid1(norm)
        hidden = self.relu1(hidden)
        hidden = self.do1(hidden)

        hidden = self.batch_norm2(hidden)
        hidden = self.linear_hid2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.do2(hidden)

        hidden = self.batch_norm3(hidden)
        output = self.linear_out(hidden)
        return output

## Define the network model
bias = True
n_hidden = 100
p_dropout = 0.5
# net = nn.Sequential(
# 		  Flatten(),
#           nn.Linear(field_size**2, n_hidden, bias=bias),
#           # nn.Linear(field_size, 2, bias=bias),
#           # nn.Dropout(),
#           # nn.ReLU(),
#           # nn.Linear(10, 10),
#           # nn.Dropout(),
#           # nn.ReLU(),
#           nn.Linear(n_hidden, n_labels, bias=bias),
#         )
net = MyNet(n_labels, field_size = field_size, n_hidden=n_hidden, bias=bias, p_dropout=p_dropout)
if b_use_cuda:
    net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

# criterion = F.mse_loss
# criterion = nn.CrossEntropyLoss()
# criterion = earth_mover_loss
criterion = earth_mover_loss2


batch_size = 100
n_iter = 2000
t_start = time.time()
history_loss_train = np.array([])
history_loss_test = np.array([])

for i_iter in range(n_iter):
    net.train()

    # Build batch
    data, targ = make_image_batch(batch_size, field_size=field_size, obj_size=obj_size)
    data, targ = Variable(data), Variable(targ)

    output = net(data)
    loss = criterion(output, targ)
    # loss = torch.mean(torch.abs(output - targ))

    loss.backward()
    optimizer.step()

    if ((i_iter+1)%100==0) | (i_iter==0):
        data, targ = make_image_batch(batch_size, field_size=field_size, obj_size=obj_size)
        data, targ = Variable(data, requires_grad=False), Variable(targ, requires_grad=False)
        net.eval()
        output = net(data)
        loss_test = criterion(output, targ)
        print('Iter %.3d: Training Loss = %.4f, Test Loss = %.4f' % (i_iter+1, loss.data.cpu().numpy()[0], loss_test.data.cpu().numpy()[0]))
        history_loss_train = np.append(history_loss_train, loss.data.cpu().numpy()[0])
        history_loss_test = np.append(history_loss_test, loss_test.data.cpu().numpy()[0])

print('Training duration: %0.4f sec' % (time.time()-t_start))

fn = 1

plt.figure(fn)
fn += 1
plt.plot(history_loss_train)
plt.plot(history_loss_test)
plt.grid(True)
plt.xlabel('Logging interval')
plt.ylabel('Loss')
plt.legend('Training data','Test data')

# Test
print('\nRunning model on test data...')
data, targ = make_image_batch(100, field_size=field_size, obj_size=obj_size)
data, targ = Variable(data, requires_grad=False), Variable(targ, requires_grad=False)
net.eval()
output = net(data)
output = F.softmax(output, dim=1)
loss = criterion(output, targ)
plt.figure(fn)
fn += 1
plt.clf()
plt.plot(targ.data.cpu().numpy(), output.data.cpu().numpy(), 'o')
plt.title('Loss = %.4f' % (loss))
v = plt.axis()
mn = min(v[0], v[2])
mx = max(v[1], v[3])
plt.axis('equal')
plt.axis([mn, mx, mn, mx])
plt.plot([mn, mx],[mn, mx],'k--')
plt.grid(True)

# For each GT target, plot a row that is the mean PMF for samples with that target
mean_pmf = np.zeros((n_labels, n_labels))
for i in range(n_labels):
    idx = np.where(targ.data.cpu().numpy()==i)[0]
    if len(idx) > 0:
        mean_pmf[i] = np.mean(output[idx].data.cpu().numpy(), axis=0)
plt.figure(fn)
fn += 1
plt.clf()
plt.imshow(mean_pmf)






