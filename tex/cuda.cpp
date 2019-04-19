// definicja kernela
__global__ void VecAdd(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    (...)
        // wywołanie kernela na N wątkach
        VecAdd<<<1, N>>>(A, B, C);
}

//------------------------------------------

// wiele bloków z wątkami
__global__ void vecAdd(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

//------------------------------------------

__global__ void printDim()
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("blockDim.x: %d \t gridDim.x: %d \n", blockDim.x, gridDim.x);
}

//------------------------------------------

using namespace std;
int main()
{
    //dane CPU
    vector<int> v{0, 1, 2, 3, 4, 5};

    //tworzenie wskaźnika dla danych GPU
    int *d_data;
    //alokowanie pamięci GPU
    cudaMalloc(&d_data, v.size() * sizeof(int));
    //kopiowanie pamięci z RAM do VRAM
    cudaMemcpy(d_data, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(32, 1, 1);
    dim3 grid_dim(5, 1, 1);
    my_kernel<<<grid_dim, block_dim>>>(d_data);

    //program czeka na skończenie obliczeń przez kartę
    cudaDeviceSynchronize();

    //kopiowanie danych z VRAM do RAM
    cudaMemcpy(v.data(), d_data, v.size() * sizeof(float), cudaMemcpyDeviceToHost);
    //zwolnienie pamięci GPU
    cudaFree(d_data);
}

//------------------------------------------

__global__ void reverse(float *A, int n)
{
    __shared__ float tmp[64];
    int i = threadIdx.x;
    int i_r = n - i - 1;
    tmp[i] = A[i];
    __syncthreads(); //bariera dla wątków w bloku
    A[i] = tmp[i_r];
}

//------------------------------------------

__global__ void sum(float *A, float *result)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(result, A[i]);
}

int main()
{
    ...
    sum<<<1, 128>>>(d_a, d_result); //OK
    sum<<<64, 2>>>(d_a, d_result);      //marnowanie zasobów
    sum<<<4, 32>>>(d_a, d_result);      //OK
    ...
}

//------------------------------------------