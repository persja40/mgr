#include <vector>
#include <cuda_runtime.h>
// template <typename T>
// __global__ xTx(T* data, int dim, T* result)

template <typename T>
__global__ T kernel_gpu(T *data, int dim)
{
    __shared__ T result = 0;
    if (threadIdx.x < dim)
    {
        atomicAdd(result, data[threadIdx.x] * 2);
    }
    return result;
}

