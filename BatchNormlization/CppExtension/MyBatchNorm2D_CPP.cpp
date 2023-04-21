// #include <iostream>
#include <vector>
// #include <torch/torch.h>
#include <torch/extension.h>


std::vector<torch::Tensor> MyBatchNorm2D_train_forward(
    torch::Tensor input, 
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor momentum, 
    torch::Tensor eps) {
  auto batch_mean = input.mean({0,2,3}, /*keepdim=*/true);
  auto batch_var = input.sub(batch_mean).pow(2).mean({0,2,3}, /*keepdim=*/true);
  auto batch_std = batch_var.add(eps.unsqueeze(-1).unsqueeze(-1)).sqrt();
  auto batch_norm = input.sub(batch_mean).div(batch_std);
  auto output = batch_norm.mul(gamma.unsqueeze(-1).unsqueeze(-1)).add(beta.unsqueeze(-1).unsqueeze(-1));
  auto new_running_mean = running_mean.mul(momentum).add(batch_mean.squeeze(-1).squeeze(-1).mul(1 - momentum));
  auto new_running_var = running_var.mul(momentum).add(batch_var.squeeze(-1).squeeze(-1).mul(1 - momentum));

  return {output, batch_norm, batch_mean, batch_var, batch_std, new_running_mean, new_running_var};
}

std::vector<torch::Tensor> MyBatchNorm2D_validation_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor eps){
  auto batch_norm = input.sub(running_mean.unsqueeze(-1).unsqueeze(-1)).div(running_var.add(eps).unsqueeze(-1).unsqueeze(-1).sqrt());
  auto output = batch_norm.mul(gamma.unsqueeze(-1).unsqueeze(-1)).add(beta.unsqueeze(-1).unsqueeze(-1));

  return {output};
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
  auto N = input.size(0);
  auto C = input.size(1);
  auto H = input.size(2);
  auto W = input.size(3);
  auto d_gamma = batch_norm.mul(grad_output).sum({0}, /*keepdim=*/true).sum({-1}, /*keepdim=*/false).sum({-1}, /*keepdim=*/false);
  auto d_beta = grad_output.sum({0},/*keepdim=*/true).sum({-1}, /*keepdim=*/false).sum({-1}, /*keepdim=*/false);
  auto d_batch_norm = grad_output.mul(gamma.unsqueeze(-1).unsqueeze(-1));
  auto d_batch_var = (d_batch_norm.mul(input.sub(batch_mean)).mul(-0.5).mul(batch_var.add(eps.unsqueeze(-1).unsqueeze(-1)).pow(-1.5))).sum({0,2,3}, /*keepdim=*/true);
  auto d_batch_mean = ((d_batch_norm.mul(-1).div(batch_std)).sum({0,2,3}, /*keepdim=*/true)).add(d_batch_var.mul((input.sub(batch_mean).mul(-2).sum({0}, /*keepdim=*/true)).div(input.size(0))));
  auto d_input = d_batch_norm.div(batch_std).add(d_batch_var.mul(2).div(N).div(H).div(W).mul(input.sub(batch_mean))).add(d_batch_mean.div(N).div(H).div(W));
  return {d_input, d_gamma, d_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("train_forward", &MyBatchNorm2D_train_forward, "MyBatchNorm1D train forward");
  m.def("validation_forward", &MyBatchNorm2D_validation_forward, "MyBatchNorm1D validation forward");
  m.def("train_backward", &MyBatchNorm2D_train_backword, "MyBatchNorm1D train backward");
}


// int main(){
//   auto input = torch::randn({4,3});
//   auto gamma = torch::ones({3});
//   auto beta = torch::zeros({3});
//   auto running_mean = torch::zeros({3});
//   auto running_var = torch::ones({3});
//   auto momentum = torch::tensor(0.1);
//   auto eps = torch::tensor(1e-5);
//   auto output = MyBatchNorm1D_train_forward(input, gamma, beta, running_mean, running_var, momentum, eps);
//   std::cout << output[0] << std::endl;  
//   std::cout << output[1] << std::endl;
//   std::cout << output[2] << std::endl;
//   std::cout << output[3] << std::endl;
//   std::cout << output[4] << std::endl;
//   std::cout << output[5] << std::endl;
//   std::cout << output[6] << std::endl;
//   return 0;

// }