#include <iostream>
#include "read_iris.h"

#include <vector>
#include "kernel.h"
using namespace std;

int main(int argc, char **argv)
{
    auto t = read_iris();
    std::cout << "Algorithm " << t.size() << std::endl;
    vector<float> test{0, 1, 2, 3, 4};
    float *d_test;
    cudaMalloc((void **)&d_test, test.size());
    cudaMemcpy(d_test, test.data(), test.size(), cudaMemcpyHostToDevice);
    std::cout << kernel_gpu<float>(d_test, test.size()) << std::endl;
    cudaFree(d_test);
}