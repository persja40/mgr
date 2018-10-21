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

__global__ void g_h_sum(float *data, int m, int n, float h, float *answer, float *sync_data)
{
    int min = m * threadIdx.x / 32;
    int max;
    if (threadIdx.x == 31) //last element <x, y> else <x, y)
        max = m * (threadIdx.x+1) / 32 + 1;
    else
        max = m * (threadIdx.x+1) / 32;
        
    float part = 0.0;
    for(int j=min; j<max; j++){
        float xTx = 0.0;
        for(int it=0; it<n; it++)
            xTx += powf((data[blockIdx.x * n + it] - data[j * n + it]) / h, 2.0);
        part += expf(-0.25 * xTx) / pow(4 * M_PI, n * 0.5) - 2 * expf(-0.5 * xTx) / (2 * pow(M_PI, n * 0.5));
    }
    // if (blockIdx.x == 0)
    //     std::printf("threadIdx: %d \t%f \t%d : %d\n", threadIdx.x, part, min, max);

    atomicAdd(&sync_data[blockIdx.x], part);
    __syncthreads();

    if (threadIdx.x == 0){
        // std::printf("BlockIdx: %d \t%f\n", blockIdx.x, sync_data[blockIdx.x]);
        atomicAdd(answer, sync_data[blockIdx.x]);
    }

    // if (threadIdx.x < n)
    // {
    //     // std::printf("Hello %d %d \n", blockIdx.x, threadIdx.x);
    //     int j = blockIdx.x % m;
    //     int i = blockIdx.x / m;
    //     float xTx = 0.0;
    //     for(int it=0; it<n; it++){
    //         xTx = ((data[threadIdx.x + j * n] - data[threadIdx.x + i * n]) / h) * ((data[threadIdx.x + j * n] - data[threadIdx.x + i * n]) / h);
    //     }
    //     xTx = expf(-0.25 * xTx) / pow(4 * M_PI, n * 0.5) - 2 * expf(-0.5 * xTx) / (2 * pow(M_PI, n * 0.5));
    //     // atomicAdd(answer, xTx);

    //     // __shared__ float *tmp_vec;
    //     // if (threadIdx.x == 0)
    //     //     cudaMalloc(&tmp_vec, n * sizeof(float));
    //     // __syncthreads();

    //     // tmp_vec[threadIdx.x] = (data[threadIdx.x + j * n] - data[threadIdx.x + i * n]) / h;
    //     // __syncthreads();

    //     // __shared__ float xTx;
    //     // if (threadIdx.x == 0)
    //     //     xTx = 0.0;
    //     // __syncthreads();

    //     // // std::printf("Hello %d %d %f\n", blockIdx.x, threadIdx.x, xTx);
    //     // // __syncthreads();

    //     // atomicAdd(&xTx, tmp_vec[threadIdx.x] * tmp_vec[threadIdx.x]);
    //     // __syncthreads();

    //     // if (threadIdx.x == 0)
    //     // {
    //     //     xTx = expf(-0.25 * xTx) / pow(4 * M_PI, n * 0.5) - 2 * expf(-0.5 * xTx) / (2 * pow(M_PI, n * 0.5));
    //     //     atomicAdd(answer, xTx);
    //     // }

    //     // __syncthreads();
    //     // if (threadIdx.x == 0){
    //     //     cudaFree(tmp_vec);
    //     //     // std::printf("Hello %d %d answer:%f\n", blockIdx.x, threadIdx.x, xTx);
    //     //     // std::printf("Hello %d %d answer:%f\n", blockIdx.x, threadIdx.x, *answer);
    //     // }
    // }
}

int main(int argc, char **argv)
{
    auto tpl = read_iris_gpu();    
    int m = std::get<1>(tpl);
    int n = std::get<2>(tpl);

    std::cout<<"Data size: "<<m<<std::endl;

    auto& t = std::get<0>(tpl);

    const float *ptr = t.data();
    // for(int i=0; i<m; i++)
    //     std::cout<<ptr[i*n]<<" "<<ptr[i*n+1]<<" "<<ptr[i*n+2]<<" "<<ptr[i*n+3]<<std::endl;

    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudeMemset(d_answer, 0, sizeof(float));

    float *d_t;
    cudaMalloc(&d_t, m * n * sizeof(float));
    cudaMemcpy(d_t, t.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    float *sync_data;
    cudaMalloc(&sync_data, m * sizeof(float));
    cudaMemset(sync_data, 0, m * sizeof(float));

    // for(int i=0; i< 1000; i++){
        g_h_sum<<<m, 32>>>(d_t, m, n, 1.0, d_answer, sync_data);
        cudaDeviceSynchronize();
    // }
    float answer = 989.123;
    cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU: " << answer << std::endl;

    cudaFree(d_t);
    cudaFree(d_answer);
}