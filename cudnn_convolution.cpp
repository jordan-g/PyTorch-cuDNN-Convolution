#include <torch/extension.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

/*
PyTorch extension enabling direct access to the following cuDNN-accelerated C++ functions
that are included in PyTorch:

    - cudnn_convolution
    - cudnn_convolution_backward_weight
    - cudnn_convolution_backward_input

The functions defined here can be called from Python in replacement of
torch.nn.conv2d, torch.nn.grad.conv2d_weight and torch.nn.grad.conv2d_input,
and run significantly faster. See 'example.py' for how these functions
are called.

Adapted from code posted by hanspinckaers:
https://discuss.pytorch.org/t/cuda-error-with-cudnn-convolution-backward-weight-function/41214
*/

at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

    return at::cudnn_convolution(
        input,
        weight,
        bias,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic); 
}

at::Tensor convolution_backward_weight(
    const at::Tensor& input,
    c10::ArrayRef<int64_t> weight_size,
    const at::Tensor& grad_output,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {

    return at::cudnn_convolution_backward_weight(
        weight_size,
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
}

at::Tensor convolution_backward_input(
    c10::ArrayRef<int64_t> input_size,
    const at::Tensor& weight,
    const at::Tensor& grad_output,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {

    return at::cudnn_convolution_backward_input(
        input_size,
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convolution", &convolution, "convolution");
    m.def("convolution_backward_weight", &convolution_backward_weight, "convolution backward weight");
    m.def("convolution_backward_input", &convolution_backward_input, "convolution backward input");
}