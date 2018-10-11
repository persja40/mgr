#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <numeric>
#include <math.h>
#include <stdio.h>

#include "read_iris.h"

/*
------------------------------------------------------------------------------------
    GPU
------------------------------------------------------------------------------------
*/

__global__ void xTx(float *data, int dim, float *answer)
{
    if (threadIdx.x < dim)
    {
        atomicAdd(answer, data[threadIdx.x] * 2);
        printf("Hello cuda thread %d\n", threadIdx.x);
    }
}

__global__ void kernel_gpu(float *data, int dim, float *answer)
{
    float *mult;
    cudaMalloc(&mult, sizeof(float));
    *mult = 0.0;
    printf("inside cuda %f\n", *mult);
    xTx<<<1, dim>>>(data, dim, mult);
    cudaDeviceSynchronize();
    printf("inside cuda %f\n", *mult);
    // expf single precision
    *answer = expf(-0.5 * (*mult)) / (2 * pow(M_PI, dim * 0.5));
    cudaFree(mult);
    printf("kernel\n");
}

/*
------------------------------------------------------------------------------------
    CPU
------------------------------------------------------------------------------------
*/

int main(int argc, char **argv)
{
    // auto t = read_iris();
    // std::cout << "Algorithm " << t.size() << std::endl;
    std::vector<float> test{0, 1, 2, 3, 4};

    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    float *d_test;
    cudaMalloc(&d_test, test.size() * sizeof(float));

    float answer = 0.0;
    cudaMemcpy(d_answer, &answer, sizeof(answer), cudaMemcpyHostToDevice);

    // answer = 123.456;
    cudaMemcpy(&answer, d_answer, sizeof(answer), cudaMemcpyDeviceToHost);
    std::cout << "SET: " << answer << std::endl;

    cudaMemcpy(d_test, test.data(), test.size() * sizeof(float), cudaMemcpyHostToDevice);

    // xTx<<<1, test.size()>>>(d_test, test.size(), d_answer);
    cudaDeviceSynchronize();

    kernel_gpu<<<1, 1>>>(d_test, test.size(), d_answer);
    cudaDeviceSynchronize();

    answer = 989.123;
    cudaMemcpy(&answer, d_answer, sizeof(answer), cudaMemcpyDeviceToHost);
    std::cout << "GPU: " << answer << std::endl;

    auto cpu_sum = [](float a, float b) {
        return a + 2 * b;
    };
    std::cout << "CPU: " << std::accumulate(begin(test), end(test), 0, cpu_sum) << std::endl;
    cudaFree(d_test);
    cudaFree(d_answer);
}