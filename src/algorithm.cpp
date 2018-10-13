#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <cmath>
#include <thread>
#include <future>
#include <mutex>

#include "read_iris.h"

float g_h_sum_cpu(std::vector<std::vector<float>> &data, float h)
{
    int m = data.size();
    int n = data[0].size();
    unsigned nr_threads = std::thread::hardware_concurrency();
    float answer = 0.0;
    std::mutex mtx;
    auto par_fun = [&](int min, int max) -> void {
        float r = 0;
        std::vector<float> tmp_vec(n, 0);
        for (int i = min; i < max; i++)
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                    tmp_vec[k] = (data[j][k] - data[i][k]) / h;
                float t = 0.0; //xTx
                for (const auto &e : tmp_vec)
                    t += e * e;
                r += exp(-0.25 * t) / pow(4 * M_PI, n * 0.5) - 2 * exp(-0.5 * t) / (2 * pow(M_PI, n * 0.5));
            }
        std::lock_guard lg(mtx);
        answer += r;
    };

    std::vector<std::future<void>> fut_handler{};

    // SHITY LAUNCH
    int tpt = static_cast<int>(std::ceil(static_cast<double>(m) / nr_threads));
    for (int i = 0; i < m; i++)
    {
        int min = i * tpt;
        int max = (i + 1) * tpt;
        if (max >= m)
            max = 0;
        fut_handler.push_back(std::async(std::launch::async, par_fun, min, max));
    }

    for (const auto &e : fut_handler) //sync threads
        e.wait();

    return answer;
}

int main(int argc, char **argv)
{
    auto t = read_iris();
    std::cout << "t size: " << t.size() << std::endl;
    std::cout << "CPU: " << g_h_sum_cpu(t, 2.0) << std::endl;
}