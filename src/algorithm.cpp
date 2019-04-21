#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <cmath>
#include <thread>
#include <future>
#include <mutex>

#include "read_iris.h"
using namespace std;

vector<float> g_h(std::vector<std::vector<float>> &data, float h)
{
    int m = data.size();
    int n = data[0].size();
    unsigned nr_threads = thread::hardware_concurrency();
    std::vector<float> answer(n, 0);
    std::mutex mtx;
    auto par_fun = [&](int min, int max) -> void {
        std::vector<float> r(n, 0);
        std::vector<float> tmp_vec(n, 0);//xTx
        for (int i = min; i < max; i++)
            for (int j = 0; j < m; j++)
            {
                // for (int k = 0; k < n; k++)
                //     cout << "i: " << i << "\tk :" << k <<"\t"<<data[i][k]<<endl;
                for (int k = 0; k < n; k++)
                    tmp_vec[k] = (data[j][k] - data[i][k]) / h;
                for (auto &e : tmp_vec)
                    e = e * e;   
                for(int k=0; k<r.size(); k++)
                    r[k] += exp(-0.25 * tmp_vec[k]) / pow(4 * M_PI, n * 0.5) - 2 * exp(-0.5 * tmp_vec[k]) / (2 * pow(M_PI, n * 0.5));
            }
        std::lock_guard lg(mtx);
        for(int i=0; i<answer.size(); i++)
            answer[i] += r[i];
    };

    vector<future<void>> fut_handler{};

    // SHITY LAUNCH
    int tpt = static_cast<int>(std::ceil(static_cast<double>(m) / nr_threads));
    // cout << "tpt: " << tpt << endl;
    for (int i = 0; i < nr_threads; i++)
    {
        int min = i * tpt;
        int max = (i + 1) * tpt;
        if (max > m)
            max = m;
        // cout << "min: " << min << "\tmax: " << max << endl;
        fut_handler.push_back(std::async(std::launch::async, par_fun, min, max));
    }

    for (const auto &e : fut_handler) //sync threads
        e.wait();    

    for(auto& e: answer)
        e = e/(pow(m,2)*pow(h, n)) + 2/(m*pow(h, n))/(2 * pow(M_PI, n * 0.5));

    return answer;
}

int main(int argc, char **argv)
{
    auto t = read_iris_cpu();
    std::cout << "Data size: " << t.size() << std::endl;
    // for (const auto v : t)
    // {
    //     for (const auto e : v)
    //         std::cout << e << " ";
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < 1000; i++)
    auto x = g_h(t, 1.0);
    for(const auto& e: x)
        std::cout<< e<< std::endl;
    // std::cout << "CPU: " << g_h_sum_cpu(t, 1.0) << std::endl;
}