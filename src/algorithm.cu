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
#include <cstring>

#include "read_iris.h"

using namespace std;

/*
------------------------------------------------------------------------------------
    GPU
------------------------------------------------------------------------------------
*/
const int THREADS_MAX = 1024;
const int WARP = 32;

__global__ void g_h_sum(float *data, int m, int n, float h, float *answer)
{
    int min = m * threadIdx.x / 32;
    int max;
    if (threadIdx.x == 31) //last element <x, y> else <x, y)
        max = m * (threadIdx.x + 1) / 32 + 1;
    else
        max = m * (threadIdx.x + 1) / 32;

    __shared__ float rr;
    if (threadIdx.x == 0)
        rr = 0;
    __syncthreads();
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

    atomicAdd(&rr, part);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        // std::printf("BlockIdx: %d \t%f\n", blockIdx.x, sync_data[blockIdx.x]);
        atomicAdd(answer, rr);
    }
}

__device__ void estimator(float *data, int m, int n, float h, float *x, float *answer)
{
    int min = m * threadIdx.x / 32;
    int max;
    if (threadIdx.x == 31) //last element <x, y> else <x, y)
        max = m * (threadIdx.x + 1) / 32 + 1;
    else
        max = m * (threadIdx.x + 1) / 32;

    __shared__ float rr;

    if (threadIdx.x == 0)
        rr = 0;
    __syncthreads();

    float result = 0.0;
    for (int j = min; j < max; j++)
    {
        float xTx = 0.0;
        for (int i = 0; i < n; i++)
            xTx += powf((x[i] - data[j * n + i]) / h, 2.0);
        result += expf(-0.5 * xTx) / (2 * pow(M_PI, n * 0.5));
    }
    atomicAdd(&rr, result);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        rr = rr / (m * powf(h, n));
        atomicAdd(answer, rr);
    }
}

__global__ void alg_step(float *data, float *newData, int m, int n, float h, float dx = 0.01)
{
    float b = powf(h, 2.0) / (n + 2);
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("h=%f \t b=%f\n", h, b);
    __shared__ float est_fdx;
    __shared__ float est_f;
    __shared__ float *xdx;
    if (threadIdx.x == 0)
    {
        est_fdx = 0.0;
        est_f = 0.0;
        cudaMalloc(&xdx, n * sizeof(float));
        memcpy(xdx, &data[blockIdx.x * n], n * sizeof(float));
        for (int i = 0; i < n; i++)
            xdx[i] += dx;
    }
    __syncthreads();
    // if(threadIdx.x == 1 && blockIdx.x == 0){
    //     for(int i=0; i<n;i++)
    //         printf("i:%d \t %f \t %f\n", i, xdx[i], data[blockIdx.x *n +i]);
    // }
    estimator(data, m, n, h, xdx, &est_fdx);
    estimator(data, m, n, h, &data[blockIdx.x * n], &est_f);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        // printf("blockIdx: %d \t est_fdx=%f \t est_f=%f\n", blockIdx.x, est_fdx, est_f);
        for (int i = 0; i < n; i++)
            newData[blockIdx.x * n + i] = data[blockIdx.x * n + i] + b * (est_fdx - est_f) / (dx * est_f);
        cudaFree(xdx);
    }
}

__global__ void distance(float *data, int m, int n, float *answer)
{
    int dataSize = m - blockIdx.x - 1; //remove itself
    int div = dataSize / WARP;
    int rest = dataSize % WARP;
    int part_size;
    __shared__ float rr;
    if (threadIdx.x == 0)
        rr = 0.0;
    __syncthreads();
    int start, stop;
    float part = 0.0;
    if (threadIdx.x < rest)
    {
        start = threadIdx.x * (div + 1);
        stop = start + (div + 1);
        part_size = div + 1;
    }
    else
    {
        start = rest * (div + 1) + abs(static_cast<int>(threadIdx.x - rest)) * div;
        stop = start + div;
        part_size = div;
    }
    if (stop < dataSize)
    {
        int offset = (blockIdx.x+1) * n;
        for (int i = 0; i < part_size; i++)
        {
            float sum_sq = 0.0;
            for(int j=0; j<n; j++)
                sum_sq += powf( data[blockIdx.x*n + j] - data[offset + (start+i)*n + j] ,2.0);
            part+= sqrt(sum_sq);
        }
    }
    atomicAdd(&rr, part);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(answer, rr);
}

float goldenRatio(float *data, int m, int n, float a = 0.000001, float b = 1'000'000, float eps = 0.0001)
{
    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));
    float l, l1, l2, f1, f2;
    const float r = 0.618034;
    while (fabs(b - a) > eps)
    {
        l = b - a;
        l1 = a + pow(r, 2) * l;
        l2 = a + r * l;

        //f1
        g_h_sum<<<m, 32>>>(data, m, n, l1, d_answer);
        cudaDeviceSynchronize();
        cudaMemcpy(&f1, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
        f1 = f1 / (pow(m, 2) * pow(l1, n)) + 2 / (m * pow(l1, n));
        cudaMemset(d_answer, 0, sizeof(float));

        //f2
        g_h_sum<<<m, 32>>>(data, m, n, l2, d_answer);
        cudaDeviceSynchronize();
        cudaMemcpy(&f2, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
        f2 = f2 / (pow(m, 2) * pow(l2, n)) + 2 / (m * pow(l2, n));
        cudaMemset(d_answer, 0, sizeof(float));

        if (f2 > f1)
            b = l2;
        else
            a = l1;
    }
    cudaFree(d_answer);
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

    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));

    float *d_t;
    cudaMalloc(&d_t, m * n * sizeof(float));
    cudaMemcpy(d_t, t.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    float *d_t2;
    cudaMalloc(&d_t2, m * n * sizeof(float));
    cudaMemcpy(d_t2, t.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_t2, 0, m * n * sizeof(float));
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

    // float *x;
    // cudaMalloc(&x, n * sizeof(float));
    // cudaMemset(x, 0, n * sizeof(float));

    // float answer = 989.123;
    // g_h_sum<<<m, 32>>>(d_t, m, n, 1.0, d_answer);
    // cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "GPU: " << answer << std::endl;
    cout << "starting golden" << endl;
    float h = goldenRatio(d_t, m, n);
    cout << "finished golden" << endl;
    alg_step<<<m, 32>>>(d_t, d_t2, m, n, h);
    cudaDeviceSynchronize();
    // float *test = new float[m * n];
    // cudaMemcpy(test, d_t, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    // float *test2 = new float[m * n];
    // cudaMemcpy(test2, d_t2, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < 4; j++)
    //         cout << test[i * n + j] << " ";
    //     cout << "\t";
    //     for (int j = 0; j < 4; j++)
    //         cout << test2[i * n + j] << " ";
    //     cout << endl;
    // }


    float answer = 989.123;
    distance<<<m, 32>>>(d_t, m, n, d_answer);
    cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU: " << answer << std::endl;
    distance<<<m, 32>>>(d_t2, m, n, d_answer);
    cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU: " << answer << std::endl;
    alg_step<<<m, 32>>>(d_t2, d_t, m, n, h);
    cudaDeviceSynchronize();
    distance<<<m, 32>>>(d_t, m, n, d_answer);
    cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU: " << answer << std::endl;

    cudaFree(d_t);
    cudaFree(d_t2);
    // cudaFree(d_answer);
}