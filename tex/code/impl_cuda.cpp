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

__global__ void estimator_array(float *data, int m, int n, float *h, float *answer){
    for(int i = blockIdx.x; i<m; i+= gridDim.x)
        estimator<<<1,4*WARP>>>(&data[i*n], data, m, n, h, &answer[i]);
}

__global__ void estimator_to_si(float *si, int size, float avg, float c){
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx<size; idx+=blockDim.x*gridDim.x)
        si[idx] = powf(si[idx]/avg, -c);
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