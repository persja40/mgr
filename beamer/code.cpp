vector<float> data;
float v = 0;
...
unsigned nr_threads = thread::hardware_concurrency();
auto reduce = [&](int thread_nr) -> float{
    float result = 0;
    for (int i = thread_nr; i < data.size(); i += nr_threads)
        result+= data[i];
    return result;
};
vector<future<float>> fut_handler{};
for (int i = 0; i < nr_threads; i++)
    fut_handler.push_back(std::async(launch::async, reduce, i));
for (auto &e : fut_handler)
    v+=e.get();




__global__ void reduce(float *data, int size, float *v){
    float part=0;
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; i<size; i+=blockDim.x*gridDim.x)
        part += data[idx];
    atomicAdd(v, part);
}
...
reduce<<<128,128>>>(data, size, v);
cudaDeviceSynchronize();


