#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// torch::Tensor elemwise_mul(const torch::Tensor& A,const torch::Tensor& B);
// __global__ void elemwise_mul_kernel(const float* A, const float* B, float* C,
//                                     const int M, const int N, const int K,
//                                     const int A_stride0, const int A_stride1,
//                                     const int B_stride0, const int B_stride1);



std::vector<torch::Tensor> MyBatchNorm1D_train_cuda_forward(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor momentum, 
    torch::Tensor eps){
    
    auto batch_mean = input.mean({0}, /*keepdim=*/true);
    auto batch_var = input.sub(batch_mean).pow(2).mean({0}, /*keepdim=*/true);
    auto batch_std = batch_var.add(eps).sqrt();
    auto batch_norm = input.sub(batch_mean).div(batch_std);
    auto output = batch_norm.mul(gamma).add(beta);
    auto new_running_mean = running_mean.mul(momentum).add(batch_mean.mul(1 - momentum));
    auto new_running_var = running_var.mul(momentum).add(batch_var.mul(1 - momentum));

    return {output, batch_norm, batch_mean, batch_var, batch_std, new_running_mean, new_running_var};
}

std::vector<torch::Tensor> MyBatchNorm1D_validation_cuda_forward(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor eps) {
    auto batch_norm = input.sub(running_mean).div(running_var.add(eps).sqrt());
    auto output = batch_norm.mul(gamma).add(beta);
    return {output};
}

std::vector<torch::Tensor> MyBatchNorm1D_train_cuda_backward(
    torch::Tensor grad_output, 
    torch::Tensor input,
    torch::Tensor batch_norm, 
    torch::Tensor batch_mean, 
    torch::Tensor batch_var, 
    torch::Tensor batch_std, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor eps) {
    auto d_gamma = batch_norm.mul(grad_output).sum({0}, /*keepdim=*/true);
    auto d_beta = grad_output.sum({0},/*keepdim=*/true);
    auto d_batch_norm = grad_output.mul(gamma);
    auto d_batch_var = (d_batch_norm.mul(input.sub(batch_mean)).mul(-0.5).mul(batch_var.add(eps).pow(-1.5))).sum({0}, /*keepdim=*/true);
    auto d_batch_mean = ((d_batch_norm.mul(-1).div(batch_std)).sum({0}, /*keepdim=*/true)).add(d_batch_var.mul((input.sub(batch_mean).mul(-2).sum({0}, /*keepdim=*/true)).div(input.size(0))));
    auto d_input = d_batch_norm.div(batch_std).add(d_batch_var.mul(2).div(input.size(0)).mul(input.sub(batch_mean))).add(d_batch_mean.div(input.size(0)));
    return {d_input, d_gamma, d_beta};
}
/*

__global__ void elemwise_mul_kernel(const float* A, const float* B, float* C,
                                    const int M, const int N, const int K,
                                    const int A_stride0, const int A_stride1,
                                    const int B_stride0, const int B_stride1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        int idx = i * N + j;

        int a_idx0 = i * A_stride0;
        int b_idx0 = j * B_stride0;

        for (int k = 0; k < K; k++) {
            int a_idx = a_idx0 + k * A_stride1;
            int b_idx = b_idx0 + k * B_stride1;

            C[idx] += A[a_idx] * B[b_idx];
        }
    }
}

torch::Tensor elemwise_mul(const torch::Tensor& A,const torch::Tensor& B) {
    int M = A.size(-2);
    int N = B.size(-1);
    int K = A.size(-1);

    TORCH_CHECK(A.dtype() == B.dtype(), "Both inputs must have the same dtype.");

    torch::Tensor output_shape;
    std::vector<int64_t> output_size;

    // Broadcasting
    if (A.dim() > B.dim()) {
        output_size = A.sizes().vec();
        output_shape = A;
        B = B.expand(output_size);
    } else {
        output_size = B.sizes().vec();
        output_shape = B;
        A = A.expand(output_size);
    }

    for (int i = 0; i < A.dim() - 2; i++) {
        if (A.size(i) != B.size(i)) {
            TORCH_CHECK(A.size(i) == 1 || B.size(i) == 1,
                        "Dimension mismatch: A.size(", i, ")=", A.size(i),
                        " does not match B.size(", i, ")=", B.size(i));
            output_size[i] = std::max(A.size(i), B.size(i));
            output_shape = at::broadcast_tensors(output_shape, output_shape.options().dtype(A.dtype()).device(A.device()), {output_size})[0];
        }
    }

    torch::Tensor C = torch::empty(output_size, A.options());

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    int A_stride0 = A.stride(-2);
    int A_stride1 = A.stride(-1);
    int B_stride0 = B.stride(-2);
    int B_stride1 = B.stride(-1);

    dim3 block_size(32, 32);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    elemwise_mul_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K,
                                                    A_stride0, A_stride1,
                                                    B_stride0, B_stride1);

    return C;
}
*/