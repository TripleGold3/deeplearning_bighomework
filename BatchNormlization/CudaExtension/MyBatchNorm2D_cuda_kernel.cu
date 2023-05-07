// #include <torch/extension.h>

// #include <cuda.h>
// #include <cuda_runtime.h>

// #include <vector>

// #include <iostream>


// // CUDA forward declarations
// namespace{
// template <typename scalar_t>
// __global__ void element_wise_mul_kernel( float* A, float* B, float* C,
//                                         int num_elements,  int num_dim,
//                                         int* A_strides,  int* B_strides,  int* C_strides,
//                                         int* A_sizes, int* B_sizes,  int* C_sizes) {
    
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < num_elements) {
//         int a_index = 0;
//         int b_index = 0;
//         int c_index = 0;

//         int offset = index;

//         for (int dim = num_dim - 1; dim >= 0; dim--) {
//             int a_size = A_sizes[dim];
//             int b_size = B_sizes[dim];
//             int c_size = C_sizes[dim];

//             int a_stride = A_strides[dim];
//             int b_stride = B_strides[dim];
//             int c_stride = C_strides[dim];

//             int i = offset % c_size;
//             offset /= c_size;

//             if (a_size == 1) {
//                 a_index += 0;
//             } else if (c_size == 1) {
//                 a_index += i * a_stride;
//             } else {
//                 a_index += (i % a_size) * a_stride;
//             }

//             if (b_size == 1) {
//                 b_index += 0;
//             } else if (c_size == 1) {
//                 b_index += i * b_stride;
//             } else {
//                 b_index += (i % b_size) * b_stride;
//             }

//             c_index += i * c_stride;
//         }

//         C[c_index] = A[a_index] * B[b_index];
//     }
// }

// } // namespace

// torch::Tensor element_wise_mul( torch::Tensor& A, torch::Tensor& B) {
//     TORCH_CHECK(A.dtype() == B.dtype(), "Both inputs must have the same dtype.");
//     TORCH_CHECK(A.device() == B.device(), "Both inputs must be on the same device.");
//     std::cout <<"A_size: " << A.sizes() << std::endl;
//     std::cout <<"B_size: " << B.sizes() << std::endl;

//     std::vector<int64_t> output_size;
//     for (int i = 0; i < A.dim(); i++) {
//         TORCH_CHECK(A.size(i) == B.size(i) || A.size(i) == 1 || B.size(i) == 1,
//                     "The size of tensor A (", A.size(i), ") must match the size of tensor B (",
//                     B.size(i), ") at non-singleton dimension ", i, ".");
//         output_size.push_back(std::max(A.size(i), B.size(i)));
//     }
//     // torch::Tensor A_broad = A.expand(output_size);
//     // torch::Tensor B_broad = B.expand(output_size);
//     torch::Tensor C = torch::empty(output_size, A.options());
//     // std::cout << "A: " << A << std::endl;
//     // std::cout << "B: " << B << std::endl;
//     // std::cout << "A_broad: " << A_broad << std::endl;
//     // std::cout << "B_broad: " << B_broad << std::endl;

//     A = A.expand(output_size);
//     B = B.expand(output_size);
//     std::cout << "A: " << A << std::endl;
//     std::cout << "B: " << B << std::endl;


//     float* d_A = A.data_ptr<float>();
//     float* d_B = B.data_ptr<float>();
//     float* d_C = C.data_ptr<float>();

//     int num_dim = A.dim();
//     int num_elements = C.numel();

//     int* A_strides = new int[num_dim];
//     int* B_strides = new int[num_dim];
//     int* C_strides = new int[num_dim];
//     int* A_sizes = new int[num_dim];
//     int* B_sizes = new int[num_dim];
//     int* C_sizes = new int[num_dim];

//     for (int i = 0; i < num_dim; i++) {
//         A_sizes[i] = A.size(i);
//         B_sizes[i] = B.size(i);
//         C_sizes[i] = C.size(i);
//     }

//     int* device_A_strides;
//     int* device_B_strides;
//     int* device_C_strides;
//     int* device_A_sizes;
//     int* device_B_sizes;
//     int* device_C_sizes;

//     cudaMalloc(&device_A_strides, sizeof(int) * num_dim);
//     cudaMalloc(&device_B_strides, sizeof(int) * num_dim);
//     cudaMalloc(&device_C_strides, sizeof(int) * num_dim);
//     cudaMalloc(&device_A_sizes, sizeof(int) * num_dim);
//     cudaMalloc(&device_B_sizes, sizeof(int) * num_dim);
//     cudaMalloc(&device_C_sizes, sizeof(int) * num_dim);

//     cudaMemcpy(device_A_strides, A_strides, sizeof(int) * num_dim, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_B_strides, B_strides, sizeof(int) * num_dim, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_C_strides, C_strides, sizeof(int) * num_dim, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_A_sizes, A_sizes, sizeof(int) * num_dim, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_B_sizes, B_sizes, sizeof(int) * num_dim, cudaMemcpyHostToDevice);
//     cudaMemcpy(device_C_sizes, C_sizes, sizeof(int) * num_dim, cudaMemcpyHostToDevice);

//     dim3 blockDim(256);
//     dim3 gridDim((num_elements + blockDim.x - 1) / blockDim.x);

//     AT_DISPATCH_FLOATING_TYPES(A.type(), "element_wise_mul_cuda", ([&] {
//         element_wise_mul_kernel<scalar_t><<<gridDim, blockDim>>>(d_A, d_B, d_C, num_elements, num_dim,
//                                                device_A_strides, device_B_strides, device_C_strides,
//                                                device_A_sizes, device_B_sizes, device_C_sizes);
//     }));
 

//     cudaFree(device_A_strides);
//     cudaFree(device_B_strides);
//     cudaFree(device_C_strides);
//     cudaFree(device_A_sizes);
//     cudaFree(device_B_sizes);
//     cudaFree(device_C_sizes);

//     return C;
// }

// std::vector<torch::Tensor> MyBatchNorm1D_train_cuda_forward(
//     torch::Tensor input, 
//     torch::Tensor gamma, 
//     torch::Tensor beta, 
//     torch::Tensor running_mean, 
//     torch::Tensor running_var, 
//     torch::Tensor momentum, 
//     torch::Tensor eps){
    
//     auto batch_mean = input.mean({0}, /*keepdim=*/true);
//     auto batch_var = input.sub(batch_mean).pow(2).mean({0}, /*keepdim=*/true);
//     auto batch_std = batch_var.add(eps).sqrt();
//     auto batch_norm = input.sub(batch_mean).div(batch_std);
//     // auto output = batch_norm.mul(gamma).add(beta);
//     // std::cout << "batch_norm: " << batch_norm << std::endl;
//     // std::cout << "gamma: " << gamma << std::endl;
//     auto output = element_wise_mul(batch_norm, gamma).add(beta);
//     auto new_running_mean = running_mean.mul(momentum).add(batch_mean.mul(1 - momentum));
//     auto new_running_var = running_var.mul(momentum).add(batch_var.mul(1 - momentum));

//     return {output, batch_norm, batch_mean, batch_var, batch_std, new_running_mean, new_running_var};
// }

// std::vector<torch::Tensor> MyBatchNorm1D_validation_cuda_forward(
//     torch::Tensor input, 
//     torch::Tensor gamma, 
//     torch::Tensor beta, 
//     torch::Tensor running_mean, 
//     torch::Tensor running_var, 
//     torch::Tensor eps) {
//     auto batch_norm = input.sub(running_mean).div(running_var.add(eps).sqrt());
//     auto output = batch_norm.mul(gamma).add(beta);
//     return {output};
// }

// std::vector<torch::Tensor> MyBatchNorm1D_train_cuda_backward(
//     torch::Tensor grad_output, 
//     torch::Tensor input,
//     torch::Tensor batch_norm, 
//     torch::Tensor batch_mean, 
//     torch::Tensor batch_var, 
//     torch::Tensor batch_std, 
//     torch::Tensor gamma, 
//     torch::Tensor beta, 
//     torch::Tensor eps) {
//     auto d_gamma = element_wise_mul(batch_norm, grad_output).sum({0}, /*keepdim=*/true);
//     auto d_beta = grad_output.sum({0},/*keepdim=*/true);
//     auto d_batch_norm = grad_output.mul(gamma);
//     auto d_batch_var = (d_batch_norm.mul(input.sub(batch_mean)).mul(-0.5).mul(batch_var.add(eps).pow(-1.5))).sum({0}, /*keepdim=*/true);
//     auto d_batch_mean = ((d_batch_norm.mul(-1).div(batch_std)).sum({0}, /*keepdim=*/true)).add(d_batch_var.mul((input.sub(batch_mean).mul(-2).sum({0}, /*keepdim=*/true)).div(input.size(0))));
//     auto d_input = d_batch_norm.div(batch_std).add(d_batch_var.mul(2).div(input.size(0)).mul(input.sub(batch_mean))).add(d_batch_mean.div(input.size(0)));
//     return {d_input, d_gamma, d_beta};
// }



#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


std::vector<torch::Tensor> MyBatchNorm2D_train_cuda_forward(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor momentum, 
    torch::Tensor eps){
    
    auto batch_mean = input.mean({0,2,3}, /*keepdim=*/true);
    auto batch_var = input.sub(batch_mean).pow(2).mean({0,2,3}, /*keepdim=*/true);
    auto batch_std = batch_var.add(eps.unsqueeze(-1).unsqueeze(-1)).sqrt();
    auto batch_norm = input.sub(batch_mean).div(batch_std);
    auto output = batch_norm.mul(gamma.unsqueeze(-1).unsqueeze(-1)).add(beta.unsqueeze(-1).unsqueeze(-1));
    auto new_running_mean = running_mean.mul(momentum).add(batch_mean.squeeze(-1).squeeze(-1).mul(1 - momentum));
    auto new_running_var = running_var.mul(momentum).add(batch_var.squeeze(-1).squeeze(-1).mul(1 - momentum));

    return {output, batch_norm, batch_mean, batch_var, batch_std, new_running_mean, new_running_var};
}

std::vector<torch::Tensor> MyBatchNorm2D_validation_cuda_forward(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor eps) {
    auto batch_norm = input.sub(running_mean.unsqueeze(-1).unsqueeze(-1)).div(running_var.add(eps).unsqueeze(-1).unsqueeze(-1).sqrt());
    auto output = batch_norm.mul(gamma.unsqueeze(-1).unsqueeze(-1)).add(beta.unsqueeze(-1).unsqueeze(-1));
    return {output};
}

std::vector<torch::Tensor> MyBatchNorm2D_train_cuda_backward(
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