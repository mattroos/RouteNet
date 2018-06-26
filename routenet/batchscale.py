import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

class BatchScale(Module):
    # Argument "momentum" is poorly named, but I'm just matching what's
    # used for BatchNorm in pytorch. It is the weighting applied to the
    # variance of the new batch when updating the ruhning variance.
    def __init__(self, num_features, eps=1e-5, momentum=0.1, linear=True):
        super(BatchScale, self).__init__()
        self.num_features = num_features
        self.linear = linear
        self.eps = eps
        self.momentum = momentum
        if linear:
            self.weight = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)
        if self.linear:
            self.weight.data.uniform_()

    def forward(self, input):
        if self.training:
            # Update variance estimate if in training mode
            batch_var = input.var(dim=0).data
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*batch_var
        return input / Variable((self.running_var + self.eps).sqrt()) * self.weight

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' linear={linear})'
                .format(name=self.__class__.__name__, **self.__dict__))


# # TODO: check contiguous in THNN
# # TODO: use separate backend functions?
# class _BatchScale(Module):

#     def __init__(self, num_features, eps=1e-5, momentum=0.1, linear=True):
#         super(_BatchScale, self).__init__()
#         self.num_features = num_features
#         self.linear = linear
#         self.eps = eps
#         self.momentum = momentum
#         if self.linear:
#             self.weight = Parameter(torch.Tensor(num_features))
#             # self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('weight', None)
#         self.register_parameter('bias', None)   # Never using a bias term
#         # self.register_buffer('running_mean', torch.zeros(num_features))
#         self.mean = torch.zeros(num_features)   # Use mean of zero, rather than running mean
#         # self.num_features = num_features
#         self.register_buffer('running_var', torch.ones(num_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         # self.running_mean.zero_()
#         self.running_var.fill_(1)
#         if self.linear:
#             self.weight.data.uniform_()
#             # self.bias.data.zero_()

#     def forward(self, input):
#         return F.batch_norm(
#             # input, self.running_mean, self.running_var, self.weight, self.bias,
#             input, self.mean, self.running_var, self.weight, self.bias,
#             # input, torch.zeros(self.num_features), self.running_var, self.weight, self.bias,
#             self.training, self.momentum, self.eps)

#     def __repr__(self):
#         return ('{name}({num_features}, eps={eps}, momentum={momentum},'
#                 ' linear={linear})'
#                 .format(name=self.__class__.__name__, **self.__dict__))


# class BatchScale1d(_BatchScale):
#     r"""Applies Batch Normalization over a 2d or 3d input that is seen as a
#     mini-batch.

#     .. math::

#         y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

#     The mean and standard-deviation are calculated per-dimension over
#     the mini-batches and gamma and beta are learnable parameter vectors
#     of size C (where C is the input size).

#     During training, this layer keeps a running estimate of its computed mean
#     and variance. The running sum is kept with a default momentum of 0.1.

#     During evaluation, this running mean/variance is used for normalization.

#     Because the BatchScale is done over the `C` dimension, computing statistics
#     on `(N, L)` slices, it's common terminology to call this Temporal BatchScale

#     Args:
#         num_features: num_features from an expected input of size
#             `batch_size x num_features [x width]`
#         eps: a value added to the denominator for numerical stability.
#             Default: 1e-5
#         momentum: the value used for the running_mean and running_var
#             computation. Default: 0.1
#         linear: a boolean value that when set to ``True``, gives the layer learnable
#             linear parameters. Default: ``True``

#     Shape:
#         - Input: :math:`(N, C)` or :math:`(N, C, L)`
#         - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

#     Examples:
#         >>> # With Learnable Parameters
#         >>> m = nn.BatchScale1d(100)
#         >>> # Without Learnable Parameters
#         >>> m = nn.BatchScale1d(100, linear=False)
#         >>> input = autograd.Variable(torch.randn(20, 100))
#         >>> output = m(input)
#     """

#     def _check_input_dim(self, input):
#         if input.dim() != 2 and input.dim() != 3:
#             raise ValueError('expected 2D or 3D input (got {}D input)'
#                              .format(input.dim()))
#         super(BatchScale1d, self)._check_input_dim(input)


# class BatchScale2d(_BatchScale):
#     r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
#     of 3d inputs

#     .. math::

#         y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

#     The mean and standard-deviation are calculated per-dimension over
#     the mini-batches and gamma and beta are learnable parameter vectors
#     of size C (where C is the input size).

#     During training, this layer keeps a running estimate of its computed mean
#     and variance. The running sum is kept with a default momentum of 0.1.

#     During evaluation, this running mean/variance is used for normalization.

#     Because the BatchScale is done over the `C` dimension, computing statistics
#     on `(N, H, W)` slices, it's common terminology to call this Spatial BatchScale

#     Args:
#         num_features: num_features from an expected input of
#             size batch_size x num_features x height x width
#         eps: a value added to the denominator for numerical stability.
#             Default: 1e-5
#         momentum: the value used for the running_mean and running_var
#             computation. Default: 0.1
#         linear: a boolean value that when set to ``True``, gives the layer learnable
#             linear parameters. Default: ``True``

#     Shape:
#         - Input: :math:`(N, C, H, W)`
#         - Output: :math:`(N, C, H, W)` (same shape as input)

#     Examples:
#         >>> # With Learnable Parameters
#         >>> m = nn.BatchScale2d(100)
#         >>> # Without Learnable Parameters
#         >>> m = nn.BatchScale2d(100, linear=False)
#         >>> input = autograd.Variable(torch.randn(20, 100, 35, 45))
#         >>> output = m(input)
#     """

#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))
#         super(BatchScale2d, self)._check_input_dim(input)


# class BatchScale3d(_BatchScale):
#     r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
#     of 4d inputs

#     .. math::

#         y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

#     The mean and standard-deviation are calculated per-dimension over
#     the mini-batches and gamma and beta are learnable parameter vectors
#     of size C (where C is the input size).

#     During training, this layer keeps a running estimate of its computed mean
#     and variance. The running sum is kept with a default momentum of 0.1.

#     During evaluation, this running mean/variance is used for normalization.

#     Because the BatchScale is done over the `C` dimension, computing statistics
#     on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchScale
#     or Spatio-temporal BatchScale

#     Args:
#         num_features: num_features from an expected input of
#             size batch_size x num_features x depth x height x width
#         eps: a value added to the denominator for numerical stability.
#             Default: 1e-5
#         momentum: the value used for the running_mean and running_var
#             computation. Default: 0.1
#         linear: a boolean value that when set to ``True``, gives the layer learnable
#             linear parameters. Default: ``True``

#     Shape:
#         - Input: :math:`(N, C, D, H, W)`
#         - Output: :math:`(N, C, D, H, W)` (same shape as input)

#     Examples:
#         >>> # With Learnable Parameters
#         >>> m = nn.BatchScale3d(100)
#         >>> # Without Learnable Parameters
#         >>> m = nn.BatchScale3d(100, linear=False)
#         >>> input = autograd.Variable(torch.randn(20, 100, 35, 45, 10))
#         >>> output = m(input)
#     """

#     def _check_input_dim(self, input):
#         if input.dim() != 5:
#             raise ValueError('expected 5D input (got {}D input)'
#                              .format(input.dim()))
#         super(BatchScale3d, self)._check_input_dim(input)
