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

float kernel(float x, int n)
{
    // cout << x <<"\t"<< exp(-0.5 * x * x) / (2 * pow(M_PI, n * 0.5)) <<endl;
    return exp(-0.5 * x * x) / (2 * pow(M_PI, n * 0.5));
}

float g_h(vector<vector<float>> &data, float h, int kth)
{
    int m = data.size();
    int n = 1; //data[0].size();
    unsigned nr_threads = thread::hardware_concurrency();
    float answer = 0;
    auto par_fun = [&](int nr) -> float {
        float r = 0;
        float xTx = 0;
        for (int i = nr; i < data.size(); i += nr_threads)
            for (int j = 0; j < m; j++)
            {
                //     cout << "i: " << i << "\tk :" << k <<"\t"<<data[i][k]<<endl;
                xTx = (data[j][kth] - data[i][kth]) / h;
                xTx = pow(xTx, 2);
                r += exp(-0.25 * xTx) / pow(4 * M_PI, n * 0.5) - 2 * exp(-0.5 * xTx) / (pow(2 * M_PI, n * 0.5));
            }
        return r;
    };

    vector<future<float>> fut_handler{};

    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, par_fun, i));

    for (auto &e : fut_handler) //sync threads
        answer += e.get();

    return answer / (pow(m, 2) * pow(h, n)) + 2 / (m * pow(h, n)) / (pow(2 * M_PI, n * 0.5));
}

float estimator(float x, vector<vector<float>> &data, vector<float> &h, float *s = nullptr)
{
    float answer = 0;
    int m = data.size();
    int n = data[0].size();
    unsigned nr_threads = thread::hardware_concurrency();
    auto par_fun = [&](int nr) -> float {
        float r = 0;
        for (int i = nr; i < data.size(); i += nr_threads)
        {
            float k = 1;
            for (int j = 0; i < n; i++)
                k *= kernel((x - data[i][j]) / h[j], n) / pow(h[j], n);
            r += k;
        }
        return r;
    };

    auto par_fun_s = [&](int nr) -> float {
        float r = 0;
        for (int i = nr; i < data.size(); i += nr_threads)
        {
            float k = 1;
            for (int j = 0; i < n; i++)
                k *= kernel((x - data[i][j]) / (h[j] * s[i]), n) / pow(h[j], n);
            r += k / pow(s[i], n);
        }
        return r;
    };

    vector<future<float>> fut_handler{};

    for (int i = 0; i < nr_threads; i++)
        if (s == nullptr)
            fut_handler.push_back(std::async(std::launch::async, par_fun, i));
        else
            fut_handler.push_back(std::async(std::launch::async, par_fun_s, i));

    for (auto &e : fut_handler) //sync threads
        answer += e.get();

    return answer / m;
}

vector<float> estimator2(vector<float> &x, vector<vector<float>> &data, vector<float> &h, float *s = nullptr)
{
    int m = data.size();
    int n = data[0].size();
    vector<float> answer(n, 0);
    // for(auto e : answer)
    //     cout<<e<<"\t";
    // cout<<endl;
    auto par_fun = [&]() -> void {
        for (int i = 0; i < data.size(); i++)
            for (int j = 0; i < n; i++){
                cout<<"IN "<<x[j]<<"\t"<<data[i][j]<<"\t"<<h[j]<<endl;
                answer[i] += kernel((x[j] - data[i][j]) / h[j], n);
                // cout << kernel((x[j] - data[i][j]) / h[j], n) << endl;
            }
    };

    auto par_fun_s = [&]() -> void {
        for (int i = 0; i < data.size(); i++)
            for (int j = 0; i < n; i++){
                answer[i] += kernel((x[j] - data[i][j]) / (h[j] * s[i]), n) / pow(s[i], n);
                // cout<< kernel((x[j] - data[i][j]) / (h[j] * s[i]), n) / pow(s[i], n) <<endl;
            }
    };

    if (s == nullptr)
        par_fun();
    else
        par_fun_s();

    // for(auto e : answer)
    //     cout<<e<<"\t";
    // cout<<endl;

    for (int i = 0; i < n; i++)
        answer[i] /= m * pow(h[i], n);

    // for(auto e : answer)
    //     cout<<e<<"\t";
    // cout<<endl<<endl;

    return answer;
}

vector<float> si_array(vector<vector<float>> &data, vector<float> &h, float c)
{
    vector<float> answer{}; //first f*
    int m = data.size();
    int n = data[0].size();
    answer.reserve(m);
    unsigned nr_threads = thread::hardware_concurrency();
    auto par_fun = [&](int nr) -> void {
        float r = 0;
        for (int i = nr; i < data.size(); i += nr_threads)
        {
            float k = 1;
            for (int j = 0; i < n; i++)
                k *= kernel(data[i][j], n) / pow(h[j], n);
            answer[i] = k / m;
        }
    };

    vector<future<void>> fut_handler{};
    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, par_fun, i));
    for (auto &e : fut_handler) //sync threads
        e.wait();

    float si_avg = accumulate(begin(answer), end(answer), 1, [&](float r, float x) {
        return pow(x, 1.0 / m);
    });

    //now si
    for (auto &e : answer)
        e = pow(e / si_avg, -c);

    return answer;
}

// API

vector<float> goldenRatio(vector<vector<float>> &data, float a_u = 0.0001, float b_u = 10'000, float eps = 0.001)
{
    std::vector<float> result{};
    for (int nth = 0; nth < data.at(0).size(); nth++)
    {
        // cout<<"n "<<nth<<endl;
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

            f1 = g_h(data, l1, nth);
            f2 = g_h(data, l2, nth);

            cout<<"l1 "<<l1<<"\tf1 "<<f1<<"\tl2 "<<l2<<"\tf2 "<<f2<<" "<<nth<<endl;

            if (f2 > f1)
                b = l2;
            else
                a = l1;
        }
        result.push_back((l1 + l2) * 0.5);
    }
    return result;
}

double distance(vector<vector<float>> &data)
{
    unsigned nr_threads = thread::hardware_concurrency();
    int n = data[0].size();
    double answer = 0;
    auto sum_d = [&](int nr) -> float {
        float r;
        for (int i = nr; i < data.size() - 1; i += nr_threads)
            for (int j = i + 1; j < data.size(); j++)
            {
                float s = 0;
                for (int k = 0; k < n; k++)
                    s += pow(data[i][k] - data[j][k], 2);
                r += sqrt(s);
            }
        return r;
    };

    vector<future<float>> fut_handler{};
    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(async(launch::async, sum_d, i));

    for (auto &e : fut_handler)
        answer += e.get();
    return answer;
}

void step(vector<vector<float>> &data, vector<float> &h, float dx = 0.01)
{
    int n = data[0].size();
    auto s = si_array(data, h, 0.5);
    unsigned nr_threads = 1;//thread::hardware_concurrency();
    vector<float> b{};
    for (auto &e : h)
        b.push_back(pow(e, 2) / (n + 2));
    auto new_data = data;

    auto par_fun = [&](int nr) -> void {
        for (int i = nr; i < data.size(); i += nr_threads)
        {
            auto r = estimator2(data[i], data, h);
            auto data_dx = data[i];
            for (auto &e : data_dx)
                e += dx;
            auto r_dx = estimator2(data_dx, data, h);
            for (int j = 0; j < n; j++)
            {
                // cout << r[j] << "\t";
                new_data[i][j] += b[j] * (r_dx[j] - r[j]) / dx / r[j];
            }
        }
    };

    vector<future<void>> fut_handler{};
    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, par_fun, i));
    for (auto &e : fut_handler) //sync threads
        e.wait();

    data = new_data;
}

void stopCondition(vector<vector<float>> &data, std::vector<float> &h, float alpha = 0.001)
{
    float d0 = distance(data);
    float dk_m1 = numeric_limits<float>::max();
    float dk = d0;
    cout << d0 << "\t" << dk_m1 << "\t" << dk << "\t" << fabs(dk_m1 - dk) << endl;
    while (fabs(dk - dk_m1) > alpha * d0)
    {
        step(data, h);
        dk_m1 = dk;
        dk = distance(data);
        cout << d0 << "\t" << dk_m1 << "\t" << dk << "\t" << fabs(dk_m1 - dk) << endl;
    }
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

    auto h = goldenRatio(t);
    for (const auto &e : h)
        std::cout << e << std::endl;
    stopCondition(t, h);
    // std::cout << "CPU: " << g_h_sum_cpu(t, 1.0) << std::endl;
}