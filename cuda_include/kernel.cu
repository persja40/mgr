#include <cuda_runtime.h>
#include <cuda.h>
// template <typename T>
// __global__ xTx(T* data, int dim, T* result)

__device__ float kernel_gpu_answer;

__global__ void kernel_gpu(float *data, int dim,float &answer)
{
    if (threadIdx.x < dim)
    {
        atomicAdd(&kernel_gpu_answer, data[threadIdx.x] * 2);
    }
}
