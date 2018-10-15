#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <numeric>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <tuple>


#include "read_iris.h"

/*
------------------------------------------------------------------------------------
    GPU
------------------------------------------------------------------------------------
*/

__global__ void g_h_sum(float *data, int m, int n, float h, float *answer)
{
    if (blockIdx.x < m * m)
        if (threadIdx.x < n)
        {
            // std::printf("Hello %d %d \n", blockIdx.x, threadIdx.x);
            int j = blockIdx.x % m;
            int i = blockIdx.x / m;
            __shared__ float *tmp_vec;
            if (threadIdx.x == 0)
                cudaMalloc(&tmp_vec, n * sizeof(float));
            __syncthreads();

            tmp_vec[threadIdx.x] = (data[threadIdx.x + j * n] - data[threadIdx.x + i * n]) / h;
            __syncthreads();

            __shared__ float xTx;
            if (threadIdx.x == 0)
                xTx = 0.0;
            __syncthreads();

            // std::printf("Hello %d %d %f\n", blockIdx.x, threadIdx.x, xTx);
            // __syncthreads();

            atomicAdd(&xTx, tmp_vec[threadIdx.x] * tmp_vec[threadIdx.x]);
            __syncthreads();

            if (threadIdx.x == 0)
            {
                xTx = expf(-0.25 * xTx) / pow(4 * M_PI, n * 0.5) - 2 * expf(-0.5 * xTx) / (2 * pow(M_PI, n * 0.5));
                atomicAdd(answer, xTx);
            }

            __syncthreads();
            if (threadIdx.x == 0){
                cudaFree(tmp_vec);
                // std::printf("Hello %d %d answer:%f\n", blockIdx.x, threadIdx.x, xTx);
                // std::printf("Hello %d %d answer:%f\n", blockIdx.x, threadIdx.x, *answer);
            }
        }
}

int main(int argc, char **argv)
{
    auto tpl = read_iris_gpu();    
    int m = std::get<1>(tpl);
    int n = std::get<2>(tpl);

    auto& t = std::get<0>(tpl);

    const float *ptr = t.data();
    // for(int i=0; i<m; i++)
    //     std::cout<<ptr[i*n]<<" "<<ptr[i*n+1]<<" "<<ptr[i*n+2]<<" "<<ptr[i*n+3]<<std::endl;

    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    float answer = 0.0;
    cudaMemcpy(d_answer, &answer, sizeof(float), cudaMemcpyHostToDevice);

    float *d_t;
    cudaMalloc(&d_t, m * n * sizeof(float));
    cudaMemcpy(d_t, t.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    g_h_sum<<<m * m, n>>>(d_t, m, n, 1.0, d_answer);
    cudaDeviceSynchronize();

    answer = 989.123;
    cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU: " << answer << std::endl;

    cudaFree(d_t);
    cudaFree(d_answer);
}