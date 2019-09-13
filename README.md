# RouteNet
Goal: Feedforward neural network that has hard gates such that features are passed from one layer to the next in a content aware fashion, reducing the overall number of computations needed for inference. There is some relation to CapsuleNets--however, those networks use iteration to perform dynamic routing that results in a single capsule target whereas the goal here has no recurrence/iteration, and routing that can features in one layer to more than one destination in subsequent layers.

# Notes
Similar ideas are finally starting to show up in the literature. E.g., 

 + [Batch-Shaped Channel Gated Networks](https://arxiv.org/abs/1907.06627)
 + [A Novel Design of Adaptive and Hierarchical Convolutional Neural Networks using Partial Reconfiguration on FPGA](https://arxiv.org/abs/1909.05653) (A different idea but still of interest.)

