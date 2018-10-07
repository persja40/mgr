#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <numeric>

#include "read_iris.h"

__global__ void kernel_gpu(float *data, int dim, float *answer)
{
    if (threadIdx.x < dim)
    {
        atomicAdd(answer, data[threadIdx.x] * 2);
    }
}

int main(int argc, char **argv)
{
    // auto t = read_iris();
    // std::cout << "Algorithm " << t.size() << std::endl;
    std::vector<float> test{0, 1, 2, 3, 4};

    float *d_answer;
    cudaMallocManaged(&d_answer, sizeof(float));
    *d_answer = 0.0;
    float *d_test;
    cudaMalloc(&d_test, test.size() * sizeof(float));

    float answer = 123.456;
    cudaMemcpy(&answer, d_answer, sizeof(answer), cudaMemcpyDeviceToHost);
    std::cout << "SET: " << answer << std::endl;

    cudaMemcpy(d_test, test.data(), test.size() * sizeof(float), cudaMemcpyHostToDevice);

    kernel_gpu<<<1, 50>>>(d_test, test.size(), d_answer);
    cudaDeviceSynchronize();

    answer = 989.123;
    cudaMemcpy(&answer, d_answer, sizeof(answer), cudaMemcpyDeviceToHost);
    std::cout << "GPU: " << answer << std::endl;
    std::cout << "CPU: " << std::accumulate(begin(test), end(test), 0) << std::endl;
    cudaFree(d_test);
}