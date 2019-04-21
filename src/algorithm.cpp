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

float g_h_sum(vector<vector<float>> &data, float h, int kth)
{
    int m = data.size();
    int n = data[0].size();
    unsigned nr_threads = thread::hardware_concurrency();
    float answer = 0;
    std::mutex mtx;
    auto par_fun = [&](int nr) -> void {
        float r = 0;
        float xTx = 0;
        for (int i = nr; i < data.size(); i+=nr_threads)
            for (int j = 0; j < m; j++)
            {
                //     cout << "i: " << i << "\tk :" << k <<"\t"<<data[i][k]<<endl;
                xTx = (data[j][kth] - data[i][kth]) / h;
                xTx = pow(xTx, 2);
                r += exp(-0.25 * xTx) / pow(4 * M_PI, n * 0.5) - 2 * exp(-0.5 * xTx) / (2 * pow(M_PI, n * 0.5));
            }
        std::lock_guard lg(mtx);
        answer += r;
    };

    vector<future<void>> fut_handler{};

    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, par_fun, i));

    for (const auto &e : fut_handler) //sync threads
        e.wait();    

    return answer/(pow(m,2)*pow(h, n)) + 2/(m*pow(h, n))/(2 * pow(M_PI, n * 0.5));
}

vector<float> goldenRatio(vector<vector<float>> &data, float a_u = 0.00001, float b_u = 10'000, float eps = 0.001){
    std::vector<float> result{};
    for (int nth = 0; nth < data.at(0).size(); nth++)
    {
        // cout<<"n "<<nth<<endl;
        // cout << "NTH: " << nth << endl;
        float a = a_u;
        float b = b_u;
        float l, l1, l2, f1, f2; //b-a  left    right   l_val   r_val
        const float r = 0.618034;
        while (abs(b - a) > eps)
        {
            l = b - a;
            l1 = a + pow(r, 2) * l;
            l2 = a + r * l;

            f1 = g_h_sum(data, l1, nth);
            f2 = g_h_sum(data, l2, nth);

            // cout<<"l1 "<<l1<<"\tf1 "<<f1<<"\tl2 "<<l2<<"\tf2 "<<f2<<" "<<nth<<endl;

            if (f2 > f1)
                b = l2;
            else
                a = l1;
        }
        result.push_back((l1+l2)*0.5);
    }
    return result;
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
    auto x = goldenRatio(t);
    for(const auto& e: x)
        std::cout<< e<< std::endl;
    // std::cout << "CPU: " << g_h_sum_cpu(t, 1.0) << std::endl;
}