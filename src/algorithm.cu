#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <numeric>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <tuple>
#include <fstream>

#include "read_iris.h"

using namespace std;

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
        max = m * (threadIdx.x + 1) / 32 + 1;
    else
        max = m * (threadIdx.x + 1) / 32;

    float part = 0.0;
    for (int j = min; j < max; j++)
    {
        float xTx = 0.0;
        for (int it = 0; it < n; it++)
            xTx += powf((data[blockIdx.x * n + it] - data[j * n + it]) / h, 2.0);
        part += expf(-0.25 * xTx) / pow(4 * M_PI, n * 0.5) - 2 * expf(-0.5 * xTx) / (2 * pow(M_PI, n * 0.5));
    }
    // if (blockIdx.x == 0)
    //     std::printf("threadIdx: %d \t%f \t%d : %d\n", threadIdx.x, part, min, max);

    atomicAdd(&sync_data[blockIdx.x], part);
    __syncthreads();

    if (threadIdx.x == 0)
    {
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

float goldenRatio(float *data, int m, int n, float a = 0.000001, float b = 1'000'000, float eps = 0.0001)
{
    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));
    float *sync_data;
    cudaMalloc(&sync_data, m * sizeof(float));
    cudaMemset(sync_data, 0, m * sizeof(float));
    float l, l1, l2, f1, f2;
    const float r = 0.618034;
    int i=0;
    while (fabs(b - a) > eps)
    {
        // cout<<i++<<endl;
        l = b - a;
        l1 = a + pow(r, 2) * l;
        l2 = a + r * l;

        //f1
        g_h_sum<<<m, 32>>>(data, m, n, l1, d_answer, sync_data);
        cudaDeviceSynchronize();
        cudaMemcpy(&f1, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
        f1 = f1 / (pow(m, 2) * pow(l1, n)) + 2 / (m * pow(l1, n));
        cudaMemset(d_answer, 0, sizeof(float));
        cudaMemset(sync_data, 0, m * sizeof(float));

        //f2
        g_h_sum<<<m, 32>>>(data, m, n, l2, d_answer, sync_data);
        cudaDeviceSynchronize();
        cudaMemcpy(&f2, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
        f2 = f2 / (pow(m, 2) * pow(l2, n)) + 2 / (m * pow(l2, n));
        cudaMemset(d_answer, 0, sizeof(float));
        cudaMemset(sync_data, 0, m * sizeof(float));

        if (f2 > f1)
            b = l2;
        else
            a = l1;
    }
    cudaFree(d_answer);
    cudaFree(sync_data);
    return l1;
}

int main(int argc, char **argv)
{
    auto tpl = read_iris_gpu();
    int m = std::get<1>(tpl);
    int n = std::get<2>(tpl);

    std::cout << "Data size: " << m << std::endl;

    auto &t = std::get<0>(tpl);

    // const float *ptr = t.data();
    // for(int i=0; i<m; i++)
    //     std::cout<<ptr[i*n]<<" "<<ptr[i*n+1]<<" "<<ptr[i*n+2]<<" "<<ptr[i*n+3]<<std::endl;

    // float *d_answer;
    // cudaMalloc(&d_answer, sizeof(float));
    // cudaMemset(d_answer, 0, sizeof(float));

    float *d_t;
    cudaMalloc(&d_t, m * n * sizeof(float));
    cudaMemcpy(d_t, t.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    // float *sync_data;
    // cudaMalloc(&sync_data, m * sizeof(float));
    // cudaMemset(sync_data, 0, m * sizeof(float));

    // float answer = 989.123;

    // ofstream ofs{"g.data"};
    // for(int i=0; i<= 10000; i++){
    //     float h = i;
    //     g_h_sum<<<m, 32>>>(d_t, m, n, h, d_answer, sync_data);
    //     cudaDeviceSynchronize();
    //     cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    //     answer = answer/(pow(m,2)*pow(h,n)) + 2/(m * pow(h,n));
    //     ofs << h << "\t" << answer << endl;
    //     cudaMemset(d_answer, 0, sizeof(float));
    //     cudaMemset(sync_data, 0, m * sizeof(float));
    //     cout<<h <<"\t"<< answer<<endl;
    // }
    // ofs.close();

    // g_h_sum<<<m, 32>>>(d_t, m, n, 1.0, d_answer, sync_data);
    // cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "GPU: " << answer << std::endl;
    
    std::cout<< "Min h:" << goldenRatio(d_t,m,n) <<std::endl;

    cudaFree(d_t);
    // cudaFree(d_answer);
}