import torch
from torch.utils.cpp_extension import load

# load the PyTorch extension
cudnn_convolution = load(name="cudnn_convolution", sources=["cudnn_convolution.cpp"], verbose=True)

# create dummy input, convolutional weights and bias
input  = torch.zeros(128, 3, 32, 32).to('cuda')
weight = torch.zeros(64, 3, 5, 5).to('cuda')
bias   = torch.zeros(64).to('cuda')

stride   = (2, 2)
padding  = (0, 0)
dilation = (1, 1)
groups   = 1

# compute the result of convolution
output = cudnn_convolution.convolution(input, weight, bias, stride, padding, dilation, groups, False, False)

# create dummy gradient w.r.t. the output
grad_output = torch.zeros(128, 64, 14, 14).to('cuda')

# compute the gradient w.r.t. the weights and input
grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, grad_output, stride, padding, dilation, groups, False, False, False)
grad_input  = cudnn_convolution.convolution_backward_input(input.shape, weight, grad_output, stride, padding, dilation, groups, False, False, False)

print(grad_weight.shape)
print(grad_input.shape)