import numpy as np
import torch
import torch.nn as nn



data_to_h1 = nn.Linear(10,10)
h1_to_h2 = nn.Linear(10,10)
h2_to_h1 = nn.Linear(10,10)
h2_to_out = nn.Linear(10,1)

batch_size = 100
data = torch.randn(batch_size, 10)
target = torch.randn(batch_size, 1)


# Use the network
h1 = data_to_h1(data)
h2 = h1_to_h2(h1)

n_iter = 3
for iter in range(n_iter):
    h1 = h2_to_h1(h2)
    h2 = h1_to_h2(h1)

output = h2_to_out(h2)


loss = torch.nn.functional.mse_loss(output, target)

loss.backward()

