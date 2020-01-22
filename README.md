# RouteNet
Goal: Feedforward neural network that has hard gates such that features are passed from one layer to the next in a content-aware fashion, reducing the overall number of computations needed for inference. There is some relation to CapsuleNets--however, those networks use iterations to perform dynamic routing from a capsule in layer i to a single target capsule in layer i+1, whereas the desired network here has no recurrence/iteration, and routing that can send features in one layer to more than one destination in subsequent layers.

# Notes
Similar ideas are finally starting to show up in the literature. E.g., 

 + [Batch-Shaped Channel Gated Networks](https://arxiv.org/abs/1907.06627)
 + [A Novel Design of Adaptive and Hierarchical Convolutional Neural Networks using Partial Reconfiguration on FPGA](https://arxiv.org/abs/1909.05653) (A different idea but still of interest.)
 + [Dynamic Convolutions: Exploiting Spatial Sparsity for Faster Inference](https://arxiv.org/abs/1912.03203)
 + [Gated Path Selection Network for Semantic Segmentation](https://arxiv.org/pdf/2001.06819.pdf)
