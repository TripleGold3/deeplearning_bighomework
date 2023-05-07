#include <vector>
#include <torch/extension.h>
std::vector<torch::Tensor> MyBatchNorm2D_train_cuda_forward(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor momentum, 
    torch::Tensor eps);

std::vector<torch::Tensor> MyBatchNorm2D_validation_cuda_forward(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor eps);

std::vector<torch::Tensor> MyBatchNorm2D_train_cuda_backward(
    torch::Tensor grad_output, 
    torch::Tensor input, 
    torch::Tensor batch_norm, 
    torch::Tensor batch_mean, 
    torch::Tensor batch_var, 
    torch::Tensor batch_std, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor eps);



// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> MyBatchNorm2D_train_forward(
        torch::Tensor input, 
        torch::Tensor gamma,
        torch::Tensor beta,
        torch::Tensor running_mean,
        torch::Tensor running_var,
        torch::Tensor momentum, 
        torch::Tensor eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);   
    CHECK_INPUT(momentum);
    CHECK_INPUT(eps);
    return MyBatchNorm2D_train_cuda_forward(input, gamma, beta, running_mean, running_var, momentum, eps);
}

std::vector<torch::Tensor> MyBatchNorm2D_validation_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor eps){
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    CHECK_INPUT(eps);
    return MyBatchNorm2D_validation_cuda_forward(input, gamma, beta, running_mean, running_var, eps);
}


std::vector<torch::Tensor> MyBatchNorm2D_train_backword(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor batch_norm,
    torch::Tensor batch_mean,
    torch::Tensor batch_var,
    torch::Tensor batch_std,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor eps) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(batch_norm);
    CHECK_INPUT(batch_mean);
    CHECK_INPUT(batch_var);
    CHECK_INPUT(batch_std);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    CHECK_INPUT(eps);
    return MyBatchNorm2D_train_cuda_backward(grad_output, input, batch_norm, batch_mean, batch_var, batch_std, gamma, beta, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("train_forward", &MyBatchNorm2D_train_forward, "MyBatchNorm2D train forward");
  m.def("validation_forward", &MyBatchNorm2D_validation_forward, "MyBatchNorm2D validation forward");
  m.def("train_backward", &MyBatchNorm2D_train_backword, "MyBatchNorm2D train backward");
}


