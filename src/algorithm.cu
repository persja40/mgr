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
#include <limits>
#include <list>
#include <utility>

#include "read_iris.h"

using namespace std;

/*
------------------------------------------------------------------------------------
    GPU
------------------------------------------------------------------------------------
*/
const int THREADS_MAX = 1024;
// __device__ int THREADS_MAX = 1024;
const int WARP = 32;

/*
n means 2nd dimension
data = 2d (n==1 -> 1d)
nth <0, n-1> chooses parameter
*/
__global__ void g_h_sum(float *data, int m, int n, int nth, float h, float *answer)
{
    int min = m * threadIdx.x / WARP;
    int max;
    if (threadIdx.x == 31) //last element <x, y> else <x, y)
        max = m * (threadIdx.x + 1) / WARP + 1;
    else
        max = m * (threadIdx.x + 1) / WARP;

    __shared__ float rr;
    if (threadIdx.x == 0)
        rr = 0;
    __syncthreads();

    float part = 0.0;
    for (int j = min; j < max; j++)
    {
        float xTx = powf((data[blockIdx.x * n + nth] - data[j * n + nth]) / h, 2.0);
        part += expf(-0.25 * xTx) / pow(4 * M_PI, 0.5) - 2 * expf(-0.5 * xTx) / (2 * pow(M_PI, 0.5));
    }
    atomicAdd(&rr, part);
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(answer, rr);
}

__device__ void estimator(float *data, int m, int n, float *h, float *x, float *answer)
{
    int min = m * threadIdx.x / WARP;
    int max;
    if (threadIdx.x == 31) //last element <x, y> else <x, y)
        max = m * (threadIdx.x + 1) / WARP + 1;
    else
        max = m * (threadIdx.x + 1) / WARP;

    __shared__ float *rr;

    if (threadIdx.x == 0)
    {
        cudaMalloc(&rr, n * sizeof(float));
        memset(rr, 0, n * sizeof(float));
    }
    __syncthreads();

    float *result;
    cudaMalloc(&result, n * sizeof(float));
    memset(result, 0, n * sizeof(float));

    for (int j = min; j < max; j++)
        for (int i = 0; i < n; i++)
        {
            float xTx = powf((x[i] - data[j * n + i]) / h[i], 2.0);
            result[i] += expf(-0.5 * xTx) / (2 * pow(M_PI, 0.5));
        }

    for (int i = 0; i < n; i++)
        atomicAdd(rr + i, result[i]);
    __syncthreads();
    cudaFree(result);

    if (threadIdx.x == 0)
    {
        cudaFree(rr);
        for (int i = 0; i < n; i++)
        {
            rr[i] = rr[i] / (m * h[i]);
            atomicAdd(answer + i, rr[i]);
        }
    }
}

__global__ void alg_step(float *data, float *newData, int m, int n, float *h, float dx = 0.01)
{
    __shared__ float *b;
    __shared__ float *est_fdx;
    __shared__ float *est_f;
    __shared__ float *xdx; // record from 0 to m-1 containing n attributes
    if (threadIdx.x == 0)
    {
        cudaMalloc(&b, n * sizeof(float));

        cudaMalloc(&est_fdx, n * sizeof(float));
        memset(est_fdx, 0, n * sizeof(float));

        cudaMalloc(&est_f, n * sizeof(float));
        memset(est_f, 0, n * sizeof(float));

        cudaMalloc(&xdx, n * sizeof(float));
        memcpy(xdx, &data[blockIdx.x * n], n * sizeof(float));
        for (int i = 0; i < n; i++)
        {
            xdx[i] += dx;
            b[i] = powf(h[i], 2.0) / (1 + 2);
        }
    }
    __syncthreads();

    estimator(data, m, n, h, xdx, est_fdx);
    estimator(data, m, n, h, &data[blockIdx.x * n], est_f);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < n; i++)
            newData[blockIdx.x * n + i] = data[blockIdx.x * n + i] + b[i] * (est_fdx[i] - est_f[i]) / (dx * est_f[i]);
        cudaFree(xdx);
        cudaFree(est_fdx);
        cudaFree(est_f);
        cudaFree(b);
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
        int offset = (blockIdx.x + 1) * n;
        for (int i = 0; i < part_size; i++)
        {
            float sum_sq = 0.0;
            for (int j = 0; j < n; j++)
                sum_sq += powf(data[blockIdx.x * n + j] - data[offset + (start + i) * n + j], 2.0);
            part += sqrt(sum_sq);
        }
    }
    atomicAdd(&rr, part);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(answer, rr);
}

__global__ void distanceArray(float *data, int m, int n, int d_size, float *d_array)
{
    int idx = blockIdx.x * THREADS_MAX + threadIdx.x;
    if (idx < d_size)
    {
        int first = 0;
        int jump = m - 1;
        int ctr = idx + 1;
        while (ctr > jump)
        {
            ctr -= jump;
            first++;
            jump--;
        }
        int last = first + ctr;

        float sum_sq = 0.0;
        for (int j = 0; j < n; j++)
            sum_sq += powf(data[first * n + j] - data[last * n + j], 2.0);
        d_array[idx] = sqrt(sum_sq);
    }
}

__global__ void distanceArrayMax(float *data, int d_size, float *sync_array, float *answer)
{
    int div = d_size / THREADS_MAX;
    int rest = d_size % THREADS_MAX;
    int start, stop;
    if (threadIdx.x < rest)
    {
        start = threadIdx.x * (div + 1);
        stop = start + (div + 1);
    }
    else
    {
        start = rest * (div + 1) + static_cast<int>(threadIdx.x - rest) * div;
        stop = start + div;
    }

    //find max idx in thread's part
    int idx = start;
    for (int i = start; i < stop; i++)
        if (data[idx] < data[i])
            idx = i;
    sync_array[threadIdx.x] = idx;

    __syncthreads();
    if (threadIdx.x == 0)
    {
        idx = 0;
        for (int i = 0; i < THREADS_MAX; i++)
            if (sync_array[idx] < sync_array[i])
                idx = i;
        *answer = data[idx];
    }
}

__global__ void sumDistance(float *data, int d_size, float *answer)
{
    int idx = blockIdx.x * THREADS_MAX + threadIdx.x;
    if (idx < d_size)
    {
        __shared__ float rr;
        if (threadIdx.x == 0)
            rr = 0.0;
        __syncthreads();
        atomicAdd(&rr, data[idx]);
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(answer, rr);
    }
}

__global__ void sumSquareDevDistance(float *data, int d_size, float EX, float *answer)
{
    int idx = blockIdx.x * THREADS_MAX + threadIdx.x;
    if (idx < d_size)
    {
        __shared__ float rr;
        if (threadIdx.x == 0)
            rr = 0.0;
        __syncthreads();
        atomicAdd(&rr, pow(data[idx] - EX, 2.0));
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(answer, rr);
    }
}

__global__ void estimatorArray(float *data, int d_size, float h_d, int n, float *answer)
{
    int idx = blockIdx.x * THREADS_MAX + threadIdx.x;
    if (idx < d_size)
    {
        float rr = 0.0;
        for (int i = 0; i < d_size; i++)
            rr += expf(-0.5 * (data[idx] - data[i]) / h_d) / pow(2 * M_PI, n * 0.5);
        answer[idx] = rr;
    }
}

__global__ void estimatorToSiArray(float *data, int d_size, float s_avg, float c = 0.5)
{
    int idx = blockIdx.x * THREADS_MAX + threadIdx.x;
    if (idx < d_size)
        data[idx] = pow(data[idx] / s_avg, -c);
}

__global__ void smallestXd(float *d, float *si, int d_size, float x_d, float h, float sigma, int *answer)
{
    if (*answer != 0)
    {
        float x = std::pow(10, blockIdx.x / 9 - 1) * (blockIdx.x - blockIdx.x / 9);
        float ds;
        switch (threadIdx.x / WARP)
        {
        case 0:
            ds = -sigma * 0.01;
            break;
        case 1:
            ds = 0.0;
            break;
        case 2:
            ds = sigma * 0.01;
            break;
        }
        __shared__ float f_m, f, f_p; //f -sigma, f, f+sigma
        if (threadIdx.x == 0)
        {
            f_m = 0.0;
            f = 0.0;
            f_p = 0.0;
        }
        __syncthreads();
        int thread_nr = threadIdx.x % WARP;

        int min = d_size * thread_nr / WARP;
        int max;
        if (thread_nr == 31) //last element <x, y> else <x, y)
            max = d_size * (thread_nr + 1) / WARP + 1;
        else
            max = d_size * (thread_nr + 1) / WARP;

        float result = 0.0;
        for (int j = min; j < max; j++)
        {
            float xTx = powf((x_d + ds - d[j]) / (h * si[j]), 2.0);
            result += (expf(-0.5 * xTx) / (2 * pow(M_PI, 1 * 0.5))) / si[j];
        }
        // ERROR !
        if (ds < 0)
            atomicAdd(&f_m, result);
        else if (ds > 0)
            atomicAdd(&f_p, result);
        else
            atomicAdd(&f, result);

        __syncthreads();
        if (threadIdx.x == 0) // *1/(m* h^n) not necessary
            if (f_m >= f || f > f_p)
                atomicExch(answer, 0);
    }
}

/********************************************************
WRAPERS
********************************************************/

std::vector<float> goldenRatio(float *data, int m, int n, float a_u = 0.000001, float b_u = 1'000'000, float eps = 0.0001)
{
    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));
    std::vector<float> result{};
    for (int nth = 0; nth < n; nth++)
    {
        cout << "NTH: " << nth << endl;
        float a = a_u;
        float b = b_u;
        float l, l1, l2, f1, f2;
        const float r = 0.618034;
        while (fabs(b - a) > eps)
        {
            l = b - a;
            l1 = a + pow(r, 2) * l;
            l2 = a + r * l;

            //f1
            g_h_sum<<<m, WARP>>>(data, m, n, nth, l1, d_answer);
            cudaDeviceSynchronize();
            cudaMemcpy(&f1, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
            f1 = f1 / (pow(m, 2) * l1) + 2 / (m * l1);
            cudaMemset(d_answer, 0, sizeof(float));

            //f2
            g_h_sum<<<m, WARP>>>(data, m, n, nth, l2, d_answer);
            cudaDeviceSynchronize();
            cudaMemcpy(&f2, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
            f2 = f2 / (pow(m, 2) * l2) + 2 / (m * l2);
            cudaMemset(d_answer, 0, sizeof(float));

            if (f2 > f1)
                b = l2;
            else
                a = l1;

            std::cout << "a: " << a << "\tb: " << b << std::endl;
            std::cout << "mod: " << fabs(b - a) << "\t" << eps << std::endl;
        }
        result.push_back(l1);
        cout << endl;
    }
    cudaFree(d_answer);
    return result;
}

void stopCondition(float *data, int m, int n, std::vector<float> &h, float alpha = 0.001)
{
    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));
    float d0, dk_m1, dk;

    dk_m1 = std::numeric_limits<float>::max();
    distance<<<m, WARP>>>(data, m, n, d_answer);
    cudaDeviceSynchronize();
    cudaMemcpy(&d0, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(d_answer, 0, sizeof(float));
    dk = d0;

    float *d_h;
    cudaMalloc(&d_h, n * sizeof(float));
    cudaMemcpy(d_h, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float *data_tmp;
    cudaMalloc(&data_tmp, m * n * sizeof(float));

    int ctr = 0;
    while (std::fabs(dk - dk_m1) > d0 * alpha)
    {
        alg_step<<<m, WARP>>>(data, data_tmp, m, n, d_h);
        cudaDeviceSynchronize();
        std::swap(data, data_tmp);

        printf("ctr: %d \t %f \n", ctr++, std::fabs(dk - dk_m1));

        dk_m1 = dk;
        cudaMemset(d_answer, 0, sizeof(float));
        distance<<<m, WARP>>>(data, m, n, d_answer);
        cudaDeviceSynchronize();
        cudaMemcpy(&dk, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(data_tmp);
    cudaFree(d_answer);
    cudaFree(d_h);
}

// float kernelDistance(float *data, int m, int n)
// {
//     std::cout << "kernelDistance\n";
//     int dist_size = m * (m - 1) / 2;

//     float *d_answer;
//     cudaMalloc(&d_answer, sizeof(float));
//     cudaMemset(d_answer, 0, sizeof(float));

//     float *d_dist;
//     cudaMalloc(&d_dist, dist_size * sizeof(float));

//     float *sync_array;
//     cudaMalloc(&sync_array, THREADS_MAX * sizeof(float));
//     cudaMemset(sync_array, 0, THREADS_MAX * sizeof(float));

//     distanceArray<<<(dist_size / THREADS_MAX) + 1, THREADS_MAX>>>(data, m, n, dist_size, d_dist);
//     cudaDeviceSynchronize();

//     float D = 0.0;

//     distanceArrayMax<<<1, THREADS_MAX>>>(d_dist, dist_size, sync_array, d_answer);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&D, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemset(d_answer, 0, sizeof(float));
//     cudaFree(sync_array);

//     std::cout << "golden D" << std::endl;
//     float h_d = goldenRatio(d_dist, dist_size, 1);

//     std::cout << h_d << "\t" << dist_size << std::endl;

//     float ex_d;
//     sumDistance<<<(dist_size / THREADS_MAX) + 1, THREADS_MAX>>>(d_dist, dist_size, d_answer);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&ex_d, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemset(d_answer, 0, sizeof(float));
//     ex_d = ex_d / dist_size;

//     float sigma_d;
//     sumSquareDevDistance<<<(dist_size / THREADS_MAX) + 1, THREADS_MAX>>>(d_dist, dist_size, ex_d, d_answer);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&sigma_d, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemset(d_answer, 0, sizeof(float));
//     sigma_d = std::sqrt(sigma_d / dist_size);

//     float *si; //current as f^*
//     cudaMalloc(&si, dist_size * sizeof(float));
//     estimatorArray<<<(dist_size / THREADS_MAX) + 1, THREADS_MAX>>>(d_dist, dist_size, h_d, n, si);
//     cudaDeviceSynchronize();
//     sumDistance<<<(dist_size / THREADS_MAX) + 1, THREADS_MAX>>>(si, dist_size, d_answer);
//     cudaDeviceSynchronize();
//     float si_avg;
//     cudaMemcpy(&si_avg, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemset(d_answer, 0, sizeof(float));
//     si_avg = si_avg / dist_size;

//     estimatorToSiArray<<<(dist_size / THREADS_MAX) + 1, THREADS_MAX>>>(si, dist_size, si_avg); //now si is si ;D
//     cudaDeviceSynchronize();

//     // Find x_d

//     std::cout << sigma_d << "\t" << si_avg << std::endl;
//     float xd;
//     int flags = 1;
//     int *flags_d;
//     cudaMalloc(&flags_d, sizeof(int));
//     cudaMemcpy(flags_d, &flags, sizeof(int), cudaMemcpyHostToDevice);
//     std::cout << "MAX DS: " << (static_cast<int>(100 * D) - 1) * sigma_d << std::endl;
//     for (float ds = 0.01; ds <= (static_cast<int>(100 * D) - 1); ds += 0.01)
//     {
//         xd = ds * sigma_d;
//         // std::cout<<xd<<std::endl;
//         smallestXd<<<7 * 9, WARP * 3>>>(d_dist, si, dist_size, xd, h_d, sigma_d, flags_d);
//         cudaDeviceSynchronize();
//         cudaMemcpy(&flags, flags_d, sizeof(int), cudaMemcpyDeviceToHost);
//         if (flags == 0)
//             break;
//         flags = 1;
//         cudaMemcpy(flags_d, &flags, sizeof(int), cudaMemcpyHostToDevice);
//     }

//     cudaFree(flags_d);
//     cudaFree(d_answer);
//     cudaFree(d_dist);
//     cudaFree(si);

//     return xd;
// }

std::vector<std::vector<std::vector<float>>> makeClusters(std::list<std::vector<float>> &l, float dist)
{
    auto d_f = [](std::vector<float> &v1, std::vector<float> &v2) {
        float s = 0;
        for (int i = 0; i < v1.size(); i++)
            s += std::abs(v1[i] - v2[i]);
        return std::sqrt(s);
    };
    std::vector<std::vector<std::vector<float>>> clusters{};
    int cluster_nr = 0;
    while (!l.empty())
    {
        clusters.push_back(std::vector<std::vector<float>>());
        clusters[cluster_nr].push_back(*begin(l));
        l.erase(begin(l));
        for (int i = 0; i < clusters[cluster_nr].size(); i++)
            for (auto it = l.begin(); it != l.end();)
            {
                if (d_f(*it, clusters[cluster_nr][i]) <= dist)
                {
                    clusters[cluster_nr].push_back(*it);
                    it = l.erase(it);
                }
                else
                    it++;
            }
        cluster_nr++;
    };
    return clusters;
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

    // float *d_t2;
    // cudaMalloc(&d_t2, m * n * sizeof(float));
    // cudaMemcpy(d_t2, t.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemset(d_t2, 0, m * n * sizeof(float));

    // ofstream ofs{"g.data"};
    // for(int i=0; i<= 10000; i++){
    //     float h = i;
    //     g_h_sum<<<m, WARP>>>(d_t, m, n, h, d_answer, sync_data);
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
    // g_h_sum<<<m, WARP>>>(d_t, m, n, 1.0, d_answer);
    // cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "GPU: " << answer << std::endl;

    cout << "starting golden" << endl;
    auto h = goldenRatio(d_t, m, n);
    cout << "finished golden" << endl;
    for (const auto &e : h)
        cout << e << endl;

    stopCondition(d_t, m, n, h);
    /*
                                    // float distance = kernelDistance(d_t, m, n);
                                    // std::cout << "Kernel distance: " << distance << std::endl;

                                    // std::cout << t[0] << "\t";

                                    // cudaMemcpy(t.data(), d_t, m * n * sizeof(float), cudaMemcpyDeviceToHost); //k* to host
                                    // std::cout << t[0] << std::endl;

                                    // std::list<std::vector<float>> l{};
                                    // for (int i = 0; i < m; i++)
                                    //     l.push_back({t[i * n], t[i * n + 1], t[i * n + 2], t[i * n + 3]});

                                    // auto clusters = makeClusters(l, 1.0f);
                                    // std::cout << "Clusters nr:" << clusters.size() << std::endl;
                                    // for (const auto &e : clusters)
                                    //     std::cout << e.size() << "\t";

                                    // std::cout << "\nFINISHED" << std::endl;
                                    // std::cout << l.empty() << std::endl;
*/

    // alg_step<<<m, WARP>>>(d_t, d_t2, m, n, h);
    // cudaDeviceSynchronize();

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

    // float answer = 989.123;
    // distance<<<m, WARP>>>(d_t, m, n, d_answer);
    // cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "GPU: " << answer << std::endl;
    // distance<<<m, WARP>>>(d_t2, m, n, d_answer);
    // cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "GPU: " << answer << std::endl;
    // alg_step<<<m, WARP>>>(d_t2, d_t, m, n, h);
    // cudaDeviceSynchronize();
    // distance<<<m, WARP>>>(d_t, m, n, d_answer);
    // cudaMemcpy(&answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "GPU: " << answer << std::endl;

    cudaFree(d_t);
    // cudaFree(d_t2);
    // cudaFree(d_answer);
}