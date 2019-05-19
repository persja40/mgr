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
#include "common.h"

using namespace std;

/*
------------------------------------------------------------------------------------
    GPU
------------------------------------------------------------------------------------
*/
const int THREADS_MAX = 1024;
const int WARP = 32;
const int dev = 0;

/*
n means 2nd dimension
data = 2d (n==1 -> 1d)
nth <0, n-1> chooses parameter
*/
__global__ void g_h_sum(float *data, int m, int n, int kth, float h, float *answer)
{
    __shared__ float rr;
    if (threadIdx.x == 0)
        rr = 0;
    __syncthreads();

    float part = 0;
    for(int i= blockIdx.x; i<m; i+=gridDim.x)
        for(int j= threadIdx.x; j<m; j+= blockDim.x){
            float xTx = powf((data[j * n + kth] - data[i * n + kth]) / h, 2.0);
            part += expf(-0.25 * xTx) / powf(4 * M_PI, 1 * 0.5) - 2 * expf(-0.5 * xTx) / powf(2 * M_PI, 1 * 0.5);
        }
    atomicAdd(&rr, part);

    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(answer, rr);
}

__global__ void g_h(float *data, int m, int n, float h){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n )
        data[idx] = data[idx]/(powf(m,2)*powf(h,1)) + 2/(m*powf(h,1))/powf(2 * M_PI,1*0.5); 
}

__global__ void distance(float *data, int m, int n, float *answer)
{
    __shared__ float rr;
    if (threadIdx.x == 0)
        rr = 0;
    __syncthreads();

    float part = 0;
    for(int i= blockIdx.x; i<m-1; i+=gridDim.x)
        for(int j= i + threadIdx.x; j<m; j+= blockDim.x){
            float d = 0;
            for(int k=0; k<n; k++)
                d+= powf( data[i*n + k] - data[j*n + k] ,2);
            part += sqrt(d);
        }
    atomicAdd(&rr, part);
    
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(answer, rr);
}

__device__ float kernel(float x)
{
    // cout << x <<"\t"<< exp(-0.5 * x * x) / (2 * pow(M_PI, n * 0.5)) <<endl;
    return exp(-0.5 * x * x);
}

__global__ void estimator( float *x, float *data, int m, int n, float *h, float *answer, float *s = nullptr)
{
    if (threadIdx.x == 0){
        // printf("%d \n",blockDim.x);
        *answer = 0;
    }
    __syncthreads();

    float part = 0;
    for(int i= threadIdx.x; i<m; i+= blockDim.x){
        float k = 1;
        for(int j=0; j<n; j++){
            float x_tmp = (x[j] - data[i*n+j]) / h[j];
            if(s)
                x_tmp = (x[j] - data[i*n+j]) / (h[j] *s[i] );
            k *= exp(-0.5 * x_tmp * x_tmp);
        }
        if(s)
            part += k / powf(2 * M_PI, n * 0.5) / powf(s[i], n) ;
        else
            part += k / powf(2 * M_PI, n * 0.5);
    }
    // if(s)
    //     printf("thread %d \t part %f \n", threadIdx.x, part);
    atomicAdd(answer, part);

    __syncthreads();
    if(threadIdx.x == 0){
        // if(s)
        // printf("answer %f\n", *answer);
        for(int i=0; i<n; i++)
            *answer /= h[i];
        *answer /= m;
    }
}

__global__ void step_alg(float *data, float *newData, int m, int n, float *h, float *b, float dx = 0.01, float *s = nullptr)
{
    
    float *r;
    float *r_dx;
    cudaMalloc(&r, sizeof(float));
    cudaMalloc(&r_dx, sizeof(float));
    float *x;
    cudaMalloc(&x, n * sizeof(float));
    for(int i = threadIdx.x; i<m; i+= blockDim.x){
        // printf("block %d \t %d\n", blockIdx.x, i);
        *r = 0;
        *r_dx = 0;
        estimator<<<1,4*WARP>>>(&data[i*n], data, m, n, h, r, s);
        cudaDeviceSynchronize();
        // printf("block %d \t r %f \n",blockIdx.x, *r);
        for(int j=0; j< n; j++)
            x[j] = data[i*n+j] + dx;
        estimator<<<1,4*WARP>>>(x, data, m, n, h, r_dx, s);
        cudaDeviceSynchronize();
        for(int j=0; j< n; j++){
            newData[i*n + j] = data[i*n+j] + b[j] * (*r_dx - *r) /dx / *r;
            //printf("%f\t",newData[i*n+j]);
        }
        // printf("block %d \t rdx %f \t r %f \n",blockIdx.x, *r_dx, *r);
    }
    cudaFree(x);
    cudaFree(r);
    cudaFree(r_dx);
}

__global__ void estimator_array(float *data, int m, int n, float *h, float *answer){
    for(int i = blockIdx.x; i<m; i+= gridDim.x)
        estimator<<<1,4*WARP>>>(&data[i*n], data, m, n, h, &answer[i]);
}

__global__ void reduce(float *data, int size, float* answer){
    float part=0;
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx<size; idx+=blockDim.x*gridDim.x)
        part+= data[idx];
    atomicAdd(answer, part);
}

__global__ void estimator_to_si(float *si, int size, float avg, float c){
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx<size; idx+=blockDim.x*gridDim.x)
        si[idx] = powf(si[idx]/avg, -c);
}


/*
    kernelDistance kernels
*/

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

__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}


__global__ void max_reduce(const float* const d_array, float* d_max, 
                                              const size_t elements)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -999999999; 

    while (gid < elements) {
        shared[tid] = max(shared[tid], d_array[gid]);
        gid += gridDim.x*blockDim.x;
        }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
            shared[tid] = max(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
      atomicMaxf(d_max, shared[0]);
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
        atomicAdd(&rr, data[idx]/d_size);
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(answer, rr);
    }
}

__global__ void sumDistanceDivide(float *data, int d_size, float *answer)
{
    int idx = blockIdx.x * THREADS_MAX + threadIdx.x;
    if (idx < d_size)
    {
        atomicAdd(answer, data[idx]/d_size);
        // printf("answer: %d \t val: %d \n", *answer, data[idx]);
    }
}

__global__ void sumSquareDevDistance(float *data, int d_size, float EX, float *answer)
{
    float part = 0;
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx<d_size; idx += blockDim.x*gridDim.x){
        part += powf(data[idx] - EX, 2.0)/(d_size-1);
    }
    atomicAdd(answer, part);
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

std::vector<float> goldenRatio(float *data, int m, int n, float a_u = 0.0001, float b_u = 10'000, float eps = 0.01)
{
    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));
    std::vector<float> result{};
    for (int nth = 0; nth < n; nth++)
    {
        // cout << "NTH: " << nth << endl;
        float a = a_u;
        float b = b_u;
        float l, l1, l2, f1, f2; //b-a  left    right   l_val   r_val
        const float r = 0.618034;
        while (fabs(b - a) > eps)
        {
            l = b - a;
            l1 = a + pow(r, 2) * l;
            l2 = a + r * l;

            //f1
            g_h_sum<<<128, 4*WARP>>>(data, m, n, nth, l1, d_answer);
            cudaDeviceSynchronize();
            cudaMemcpy(&f1, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
            f1 = f1 / (pow(m, 2) * pow(l1,1)) + 2 / (m * pow(l1,1)) /powf(2 * M_PI,1*0.5);
            cudaMemset(d_answer, 0, sizeof(float));

            //f2
            g_h_sum<<<128, 4*WARP>>>(data, m, n, nth, l2, d_answer);
            cudaDeviceSynchronize();
            cudaMemcpy(&f2, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
            f2 = f2 / (pow(m, 2) * pow(l2,1)) + 2 / (m * pow(l2,1))/powf(2 * M_PI,1*0.5);
            cudaMemset(d_answer, 0, sizeof(float));

            cout<<"l1 "<<l1<<"\tf1 "<<f1<<"\tl2 "<<l2<<"\tf2 "<<f2<<"\t"<<nth<<endl;

            if (f2 > f1)
                b = l2;
            else
                a = l1;

            // std::cout << "a: " << a << "\tb: " << b << std::endl;
            // std::cout << "mod: " << fabs(b - a) << "\t" << eps << std::endl;
        }
        result.push_back((l1+l2)*0.5);
        // cout << endl;
    }
    cudaFree(d_answer);
    return result;
}

void step(float *data, float *data_tmp, int m, int n, float *h, float *b, float dx = 0.1){
    float *d_si;
    cudaMalloc(&d_si, m * sizeof(float));
    cudaMemset(d_si, 0, m * sizeof(float));
    estimator_array<<<128, 1>>>(data, m, n, h, d_si);
    cudaDeviceSynchronize();
    // vector<float> si(m,0);
    // cudaMemcpy(si.data(), d_si, m*sizeof(float), cudaMemcpyDeviceToHost);
    
    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));
    // float si_avg = accumulate(begin(si), end(si), 0, [&](float r, float x) {
    //     return r + log(x);
    // });
    reduce<<<128, 4*WARP>>>(d_si, m, d_answer);
    float si_avg;
    cudaDeviceSynchronize();
    cudaMemcpy(&si_avg, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    si_avg = expf(si_avg / m);

    //now si
    float c = 0.5;
    // for (auto &e : si)
    //     e = pow(e / si_avg, -c);
    estimator_to_si<<<128,4*WARP>>>(d_si, m, si_avg, c);
    cudaDeviceSynchronize();

    // for(auto e:si)
    //     cout<<e<<"\t";
    //     cout<<endl;
    // throw int(5);
    // cudaMemcpy(d_si, si.data(), m*sizeof(float), cudaMemcpyHostToDevice);
    step_alg<<<1,128>>>(data, data_tmp, m, n, h, b, dx, d_si);
    cudaDeviceSynchronize();
    // std::vector<float> tt(m*n,0);
    // cudaMemcpy(tt.data(), data_tmp, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i=0; i<m; i++){
    //     for(int j=0; j<n;j++)
    //         cout<<tt[i*n+j]<<"\t";
    //     cout<<endl;
    // }
    // throw int(5);
    cudaFree(d_si);
    cudaFree(d_answer);
}

void stopCondition(float *data, int m, int n, std::vector<float> &h, float alpha = 0.001)
{
    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));
    
    float d0;
    distance<<<128, 4*WARP>>>(data, m, n, d_answer);
    cudaDeviceSynchronize();
    cudaMemcpy(&d0, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(d_answer, 0, sizeof(float));
    
    float dk_m1 = std::numeric_limits<float>::max();
    float dk = d0;

    float *d_h;
    cudaMalloc(&d_h, h.size() * sizeof(float));
    cudaMemcpy(d_h, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);

    float *data_tmp;
    cudaMalloc(&data_tmp, m * n * sizeof(float));

    vector<float> b{};
    for (auto &e : h)
        b.push_back(pow(e, 2) / (n + 2));
    float *d_b;
    cudaMalloc(&d_b, h.size() * sizeof(float));
    cudaMemcpy(d_b, b.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);

    cout <<"STOP " <<d0 << "\t" << dk_m1 << "\t" << dk << "\t" << fabs(dk_m1 - dk) << endl;
    while (std::fabs(dk - dk_m1) > d0 * alpha)
    {
        step(data, data_tmp, m, n, d_h, d_b);
        cudaMemcpy(data,data_tmp, m*n*sizeof(float), cudaMemcpyDeviceToDevice);
        // std::swap(data, data_tmp);

        // printf("ctr: %d \t %f \n", ctr++, std::fabs(dk - dk_m1));

        dk_m1 = dk;
        cudaMemset(d_answer, 0, sizeof(float));
        distance<<<128, 4*WARP>>>(data, m, n, d_answer);
        cudaDeviceSynchronize();
        cudaMemcpy(&dk, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
        cout << d0 << "\t" << dk_m1 << "\t" << dk << "\t" << fabs(dk_m1 - dk) << endl;
    }
    cudaFree(data_tmp);
    cudaFree(d_answer);
    cudaFree(d_h);
    cudaFree(d_b);
}

float kernelDistance(float *data, int m, int n)
{
    int dist_size = m * (m - 1) / 2;
    std::cout << "kernelDistance "<<dist_size<<endl;

    float *d_answer;
    cudaMalloc(&d_answer, sizeof(float));
    cudaMemset(d_answer, 0, sizeof(float));

    float *d_dist;
    cudaMalloc(&d_dist, dist_size * sizeof(float));

    distanceArray<<<(dist_size / THREADS_MAX) + 1, THREADS_MAX>>>(data, m, n, dist_size, d_dist);
    cudaDeviceSynchronize();

    vector<float> dist_a(dist_size, 0);
    cudaMemcpy(dist_a.data(), d_dist, dist_size*sizeof(float), cudaMemcpyDeviceToHost);
    // for(auto &e: dist_a)
    //     cout<<e<<endl;
    // cout<<endl;
    // cout<<accumulate(begin(dist_a), end(dist_a), 0)<<endl;
    // throw int(5);
    float D = 0.0;

    // distanceArrayMax<<<1, THREADS_MAX>>>(d_dist, dist_size, sync_array, d_answer);
    max_reduce<<<128,4*WARP,4*WARP*sizeof(float)>>>(d_dist,d_answer,dist_size);
    cudaDeviceSynchronize();
    cudaMemcpy(&D, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(d_answer, 0, sizeof(float));

    std::cout << "golden D" << std::endl;
    float h = goldenRatio(d_dist, dist_size, 1)[0]; // one element vector - one dimension

    std::cout <<"D "<<D<<"\t h " << h << "\t size " << dist_size << std::endl;

    float ex_d = accumulate(dist_a.begin(), dist_a.end(), 0.0)/dist_a.size();
    // sumDistavector<float> dist_a(dist_size, 0);
    // nce<<<(dist_size / THREADS_MAX) + 1, THREADS_MAX>>>(d_dist, dist_size, d_answer);
    // cudaDeviceSynchronize();
    // cudaMemcpy(&ex_d, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemset(d_answer, 0, sizeof(float));

    cout<<"EX_D: "<<ex_d<<endl;

    float sigma_d;
    sumSquareDevDistance<<<128, 4*WARP>>>(d_dist, dist_size, ex_d, d_answer);
    cudaDeviceSynchronize();
    cudaMemcpy(&sigma_d, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(d_answer, 0, sizeof(float));
    sigma_d = std::sqrt(sigma_d);

    // h = 0.246597;
    float* d_h;
    cudaMalloc(&d_h, sizeof(float));
    cudaMemcpy(d_h, &h, sizeof(float), cudaMemcpyHostToDevice);

    float *d_si; //current as f^*
    cudaMalloc(&d_si, dist_size * sizeof(float));
    estimator_array<<<128, 1>>>( d_dist ,dist_size, 1, d_h, d_si);
    cudaDeviceSynchronize();
    // vector<float> si(dist_size,0);
    // cudaMemcpy(si.data(), d_si, dist_size*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaMemset(d_answer, 0, sizeof(float));
    // float si_avg = accumulate(begin(si), end(si), 0, [&](float r, float x) {
    //     return r + log(x);
    // });
    reduce<<<128, 4*WARP>>>(d_si, dist_size, d_answer);
    float si_avg;
    cudaDeviceSynchronize();
    cudaMemset(d_answer, 0, sizeof(float));
    cudaMemcpy(&si_avg, d_answer, sizeof(float), cudaMemcpyDeviceToHost);
    si_avg = expf(si_avg / dist_size);

    //now si
    float c = 0.5;
    // for (auto &e : si)
    //     e = pow(e / si_avg, -c);

    estimator_to_si<<<128,4*WARP>>>(d_si, dist_size, si_avg, c);
    cudaDeviceSynchronize();
    // for(auto e:si)
    //     cout<<e<<"\t";
    //     cout<<endl;
    // throw int(5);
    // cudaMemcpy(d_si, si.data(), dist_size*sizeof(float), cudaMemcpyHostToDevice);

    // Find x_d
    // vector<float> s(dist_size, 0);
    // cudaMemcpy(s.data(), d_si, dist_size*sizeof(float), cudaMemcpyDeviceToHost);
    // cout<<accumulate(begin(s), end(s),0)<<endl;
    // throw int(5);
    std::cout << "SIGMA\tSI_AVG"<<endl;
    std::cout << sigma_d << "\t" << si_avg << std::endl;
    float xd, xd_ps, xd_ms;
    int flags = 1;
    int *flags_d;
    cudaMalloc(&flags_d, sizeof(int));
    cudaMemcpy(flags_d, &flags, sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "MAX DS: " << (static_cast<int>(100 * D) - 1) * sigma_d << std::endl;
    float *d_xd, *d_xd_ps, *d_xd_ms;
    cudaMalloc(&d_xd, sizeof(float));
    cudaMalloc(&d_xd_ps, sizeof(float));
    cudaMalloc(&d_xd_ms, sizeof(float));

    float *d_r1, *d_r2, *d_r3;
    cudaMalloc(&d_r1, sizeof(float));
    cudaMalloc(&d_r2, sizeof(float));
    cudaMalloc(&d_r3, sizeof(float));

    cudaStream_t st1, st2, st3;
    cudaStreamCreate(&st1);
    cudaStreamCreate(&st2);
    cudaStreamCreate(&st3);

    for (float ds = 0.01; ds <= (static_cast<int>(100 * D) - 1); ds += 0.01)
    {
        cudaMemset(d_r1, 0, sizeof(float));
        cudaMemset(d_r2, 0, sizeof(float));
        cudaMemset(d_r3, 0, sizeof(float));
        xd = ds * sigma_d;
        // if(xd>2)
        //     throw int(5);
        xd_ps = xd + 0.01*sigma_d;
        xd_ms = xd - 0.01*sigma_d;
        cudaMemcpy(d_xd, &xd, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xd_ps, &xd_ps, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xd_ms, &xd_ms, sizeof(float), cudaMemcpyHostToDevice);
        estimator<<<1,4*WARP,0,st1>>>(d_xd, d_dist, dist_size, 1, d_h, d_r1, nullptr);
        estimator<<<1,4*WARP,0,st2>>>(d_xd_ps, d_dist, dist_size, 1, d_h, d_r2, nullptr);
        estimator<<<1,4*WARP,0,st3>>>(d_xd_ms, d_dist, dist_size, 1, d_h, d_r3, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(&xd, d_r1, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&xd_ps, d_r2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&xd_ms, d_r3, sizeof(float), cudaMemcpyDeviceToHost);
        cout<<ds * sigma_d<<"\t"<<xd<<"\t"<<xd_ms<<"\t"<<xd_ps<<endl;
        if( xd < xd_ms && xd <= xd_ps ){
            xd = ds*sigma_d;
            break;
        }
        // smallestXd<<<7 * 9, WARP * 3>>>(d_dist, d_si, dist_size, xd, h, sigma_d, flags_d);
        // cudaDeviceSynchronize();
        // cudaMemcpy(&flags, flags_d, sizeof(int), cudaMemcpyDeviceToHost);
        // if (flags == 0)
        //     break;
        // flags = 1;
        // cudaMemcpy(flags_d, &flags, sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaFree(flags_d);
    cudaFree(d_answer);
    cudaFree(d_dist);
    cudaFree(d_si);
    cudaFree(d_r1);
    cudaFree(d_r2);
    cudaFree(d_r3);
    cudaFree(d_xd);
    cudaFree(d_xd_ps);
    cudaFree(d_xd_ms);

    return xd;
}

int main(int argc, char **argv)
{
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, 0);
    // WARP = deviceProp.warpSize;

    auto tpl = read_iris_gpu();
    int m = std::get<1>(tpl);
    int n = std::get<2>(tpl);

    std::cout << "Data size: " << m << std::endl;

    auto &t = std::get<0>(tpl);

    // const float *ptr = t.data();
    // for(int i=0; i<m; i++)
    //     std::cout<<ptr[i*n]<<" "<<ptr[i*n+1]<<std::endl;

    float *d_answer;
    cudaMalloc(&d_answer, n * sizeof(float));
    cudaMemset(d_answer, 0, n * sizeof(float));

    float *d_t;
    cudaMalloc(&d_t, m * n * sizeof(float));
    cudaMemcpy(d_t, t.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cout << "starting GOLDEN" << endl;
    auto h = goldenRatio(d_t, m, n);
    cout << "finished GOLDEN" << endl;
    for (const auto &e : h)
        cout <<"h\t"<< e << endl;

    stopCondition(d_t, m, n, h);

    cudaMemcpy(t.data(), d_t, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i=0; i<m; i++)
    //     std::cout<<t[i*n]<<" "<<t[i*n+1]<<" "<<t[i*n+2]<<" "<<t[i*n+3]<<std::endl;

    // cout<<"starting STOPCondition" << endl;
    // stopCondition(d_t, m, n, h);
    // cout<<"finished STOPCondition" << endl;

    float distance = kernelDistance(d_t, m, n);
    std::cout << "Kernel distance: " << distance << std::endl;

    // t[0] = 0;
    // std::cout << t[0] << "\t";

    // cudaMemcpy(t.data(), d_t, m * n * sizeof(float), cudaMemcpyDeviceToHost); //k* to host
    // std::cout << t[0] << std::endl;

    std::list<std::vector<float>> l{};
    for (int i = 0; i < m; i++)
        l.push_back({t[i * n], t[i * n + 1], t[i * n + 2], t[i * n + 3]});

    auto clusters = makeClusters(l, distance*2);
    std::cout << "Clusters nr:" << clusters.size() << std::endl;
    for (const auto &e : clusters)
        std::cout << e.size() << "\t";
    cout<<endl;

    // std::cout << "\nFINISHED" << std::endl;
    // std::cout << l.empty() << std::endl;


////////////////////////////COMMENT




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