#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <numeric>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <thread>
#include <future>
#include <atomic>

#include "read_iris.h"

/*
------------------------------------------------------------------------------------
    GPU
------------------------------------------------------------------------------------
*/

__global__ void g_h_sum(float *data, int m, int n, float h, float *answer)
{
    std::printf("Hello %d %d \n", blockIdx.x, threadIdx.x);
    if (blockIdx.x < m * m)
    {
        int j = blockIdx.x % m;
        int i = blockIdx.x / m;
        float *tmp_vec;
        cudaMalloc(&tmp_vec, n * sizeof(float));
        if (threadIdx.x < n)
            tmp_vec[threadIdx.x] = (data[threadIdx.x + j * n] - data[threadIdx.x + i * n]) / h;

        float xTx = 0.0;
        __syncthreads();

        if (threadIdx.x < n)
            atomicAdd(&xTx, tmp_vec[threadIdx.x] * tmp_vec[threadIdx.x]);
        __syncthreads();

        if (threadIdx.x == 0)
        {
            xTx = expf(-0.25 * xTx) / pow(4 * M_PI, n * 0.5) - 2 * expf(-0.5 * xTx) / (2 * pow(M_PI, n * 0.5));
            atomicAdd(answer, xTx);
        }
        cudaFree(tmp_vec);
    }
}

int main(int argc, char **argv)
{
    auto t = read_iris();
    std::cout << "t size: " << t.size() << std::endl;
    // std::vector<float> test{0, 1, 2, 3, 4};
    int m = t.size();
    int n = t[0].size();

    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    float answer = 0.0;
    cudaMemcpy(d_answer, &answer, sizeof(float), cudaMemcpyHostToDevice);

    float *d_t;
    cudaMalloc(&d_t, m * n * sizeof(float));
    cudaMemcpy(d_t, t.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    g_h_sum<<<m * m, n>>>(d_t, m, n, 2.0, d_answer);
    cudaDeviceSynchronize();

    answer = 989.123;
    cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU: " << answer << std::endl;

    cudaFree(d_t);
    cudaFree(d_answer);
}