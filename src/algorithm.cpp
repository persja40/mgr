#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>

#include "read_iris.h"
#include "kernel.cu"

int main(int argc, char **argv)
{
    auto t = read_iris();
    std::cout << "Algorithm " << t.size() << std::endl;
    std::vector<float> test{0, 1, 2, 3, 4};
    float *d_test;
    cudaMalloc((void **)&d_test, test.size());
    cudaMemcpy(d_test, test.data(), test.size(), cudaMemcpyHostToDevice);
    kernel_gpu<<<500, 1>>>(d_test, test.size(), &kernel_gpu_answer);
    cudaDeviceSynchronize();
    float answer;
    cudaMemcpyFromSymbol(&answer, "kernel_gpu_answer", sizeof(answer), 0, cudaMemcpyFromDeviceToHost);
    std::cout << answer << std::endl;
    cudaFree(d_test);
}