#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <cmath>
#include <thread>
#include <future>
#include <mutex>

#include "read_iris.h"
#include "common.h"
using namespace std;

float kernel(float x)
{
    // cout << x <<"\t"<< exp(-0.5 * x * x) / (2 * pow(M_PI, n * 0.5)) <<endl;
    return exp(-0.5 * x * x);
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

float estimator(vector<float> &x, vector<vector<float>> &data, vector<float> &h, float *s = nullptr)
{
    int m = data.size();
    int n = data[0].size();
    float answer = 0;
    // for(auto e : answer)
    //     cout<<e<<"\t";
    // cout<<endl;
    auto par_fun = [&]() -> float {
        float r = 0;
        for (int i = 0; i < data.size(); i++)
        {
            float k = 1;
            for (int j = 0; j < n; j++)
            {
                // cout << "IN " << x[j] << "\t" << data[i][j] << "\t" << h[j] << endl;
                k *= kernel((x[j] - data[i][j]) / h[j]);
                // cout << kernel((x[j] - data[i][j]) / h[j], n) << endl;
            }
            r += k / powf(2 * M_PI, n * 0.5);
        }
        return r;
    };

    auto par_fun_s = [&]() -> float {
        float r = 0;
        for (int i = 0; i < data.size(); i++)
        {
            float k = 1;
            for (int j = 0; j < n; j++)
            {
                k *= kernel((x[j] - data[i][j]) / (h[j] * s[i]));
                // cout<< kernel((x[j] - data[i][j]) / (h[j] * s[i]), n) / pow(s[i], n) <<endl;
            }
            r += k / powf(2 * M_PI, n * 0.5) / pow(s[i], n);
        }
        return r;
    };

    if (s == nullptr)
        answer = par_fun();
    else
        answer = par_fun_s();

    // for(auto e : answer)
    //     cout<<e<<"\t";
    // cout<<endl;

    for (int i = 0; i < n; i++)
        answer *= 1.0 / h[i];
    return answer / m;
}

vector<float> si_array(vector<vector<float>> &data, vector<float> &h, float c)
{
    int m = data.size();
    int n = data[0].size();
    vector<float> answer(m, 0); //first f*
    unsigned nr_threads = thread::hardware_concurrency();
    auto par_fun = [&](int nr) -> void {
        float r = 0;
        for (int i = nr; i < data.size(); i += nr_threads)
        {
            answer[i] = estimator(data[i], data, h);
        }
    };

    vector<future<void>> fut_handler{};
    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, par_fun, i));
    for (auto &e : fut_handler) //sync threads
        e.wait();

    float si_avg = accumulate(begin(answer), end(answer), 0, [&](float r, float x) {
        return r + log(x);
    });
    si_avg = expf(si_avg / m);

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

            cout << "l1 " << l1 << "\tf1 " << f1 << "\tl2 " << l2 << "\tf2 " << f2 << " " << nth << endl;

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

void step(vector<vector<float>> &data, vector<float> &h, float dx = 0.1)
{
    int n = data[0].size();
    auto s = si_array(data, h, 0.5);
    // for(auto e:s)
    //     cout<<e<<"\t";
    //     cout<<endl;
    // throw int(5);
    unsigned nr_threads = thread::hardware_concurrency();
    vector<float> b{};
    for (auto &e : h)
        b.push_back(pow(e, 2) / (n + 2));
    auto new_data = data;

    auto par_fun = [&](int nr) -> void {
        for (int i = nr; i < data.size(); i += nr_threads)
        {
            auto r = estimator(data[i], data, h, s.data());
            auto data_dx = data[i];
            for (auto &e : data_dx)
                e += dx;
            auto r_dx = estimator(data_dx, data, h, s.data());
            //cout<<"r_dx "<<r_dx<<"\t r "<<r<<endl;
            for (int j = 0; j < n; j++)
            {
                // cout << r[j] << "\t";
                new_data[i][j] += b[j] * (r_dx - r) / dx / r;
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
    cout << "STOP " << d0 << "\t" << dk_m1 << "\t" << dk << "\t" << fabs(dk_m1 - dk) << endl;
    while (fabs(dk - dk_m1) > alpha * d0)
    {
        step(data, h);
        // for (const auto v : data)
        // {
        //     for (const auto e : v)
        //         std::cout << e << " ";
        //     std::cout << std::endl;
        // }
        // throw int(5);
        dk_m1 = dk;
        dk = distance(data);
        cout << d0 << "\t" << dk_m1 << "\t" << dk << "\t" << fabs(dk_m1 - dk) << endl;
    }
}

vector<vector<float>> distanceArray(vector<vector<float>> &data)
{
    int m = data.size();
    int n = data[0].size();
    vector<vector<float>> result{};
    result.reserve(m * (m - 1) / 2);
    unsigned nr_threads = thread::hardware_concurrency();
    auto fun_dist = [&](int nr) -> vector<vector<float>> {
        vector<vector<float>> r{};
        r.reserve(m * (m - 1) / 2 / nr_threads);
        for (int i = nr; i < m; i += nr_threads)
        {
            for (int j = i + 1; j < m; j++)
            {
                float v = 0;
                for (int k = 0; k < n; k++)
                    v += pow(data[i][k] - data[j][k], 2);
                r.push_back({sqrt(v)});
            }
        }
        return r;
    };

    vector<future<vector<vector<float>>>> fut_handler{};
    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, fun_dist, i));
    for (auto &e : fut_handler)
    {
        auto w = e.get();
        result.insert(result.end(), w.begin(), w.end());
    }
    return result;
}

float maxDistance(vector<vector<float>> &data)
{
    float result = numeric_limits<float>::min();
    unsigned nr_threads = thread::hardware_concurrency();
    auto fun_max = [&](int nr) -> float {
        float r = numeric_limits<float>::min();
        for (int i = nr; i < data.size(); i += nr_threads)
            r = max(r, data[i][0]);
        return r;
    };
    vector<future<float>> fut_handler{};
    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, fun_max, i));
    for (auto &e : fut_handler)
        result = max(result, e.get());
    return result;
}

float sigma(vector<vector<float>> &data)
{
    int m = data.size();
    float result = 0; //EX
    unsigned nr_threads = thread::hardware_concurrency();
    auto fun_avg = [&](int nr) -> float {
        float r = 0;
        for (int i = nr; i < m; i += nr_threads)
            r += data[i][0] / m;
        return r;
    };
    vector<future<float>> fut_handler{};
    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, fun_avg, i));
    for (auto &e : fut_handler)
        result += e.get();
    fut_handler.clear();
    auto fun_sigma_sum = [&](int nr, float EX) -> float {
        float r = 0;
        for (int i = nr; i < m; i += nr_threads)
            r += pow(data[i][0] - EX, 2) / (m - 1);
        return r;
    };
    for (int i = 0; i < nr_threads; i++)
        fut_handler.push_back(std::async(std::launch::async, fun_sigma_sum, i, result));
    result = 0; //sigma_d
    for (auto &e : fut_handler)
        result += e.get();
    return sqrt(result);
}

bool distanceCondition(float x, float dx, vector<vector<float>> &data, float *si, vector<float> &h)
{
    vector<float> xd{x};
    auto f = estimator(xd, data, h, si);

    vector<float> xd_p_sigma{x + dx};
    auto f_p_sigma = estimator(xd_p_sigma, data, h, si);

    vector<float> xd_m_sigma{x - dx};
    auto f_m_sigma = estimator(xd_m_sigma, data, h, si);
    // cout << f_m_sigma << "\t" << f << "\t" << f_p_sigma << "\t" << x << endl;
    cout<<x<<"\t"<<f<<"\t"<<f_m_sigma<<"\t"<<f_p_sigma<<endl;
    return f_m_sigma > f && f <= f_p_sigma;
}

float clusterDistance(vector<vector<float>> &data)
{
    auto data_dist = distanceArray(data);
    cout << "Dist arr size: " << data_dist.size() << endl;
    // for(auto &e: data_dist)
    //         cout<<e[0]<<endl;
    // cout<<endl;
    // cout<<accumulate(begin(data_dist), end(data_dist), 0, [](float r, vector<float> v){return r+v[0];})<<endl;
    // throw int(5);
    float D = maxDistance(data_dist);
    cout << "D " << D << endl;
    float sigma_d = sigma(data_dist);
    cout << "Sigma_d: " << sigma_d << endl;
    auto h = goldenRatio(data_dist);
    cout << "h_d: " << h[0] << endl;
    auto si = si_array(data_dist, h, 0.5);
    // cout<<accumulate(begin(si), end(si),0)<<endl;
    // throw int(5);
    float xd = 0;
    for (xd = 0.01 * sigma_d; xd < static_cast<int>(100 * D - 1) * 0.01 * sigma_d; xd += 0.01 * sigma_d)
    {
        if (distanceCondition(xd, 0.01 * sigma_d, data_dist, si.data(), h))
            return xd;
    }
    return xd;
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
    float dist = clusterDistance(t);
    cout << "ClusterDistance: " << dist << endl;
    // std::cout << "CPU: " << g_h_sum_cpu(t, 1.0) << std::endl;

    std::list<std::vector<float>> l{};
    for (int i = 0; i < t.size(); i++)
        l.push_back(t[i]);

    // for (const auto v : t)
    // {
    //     for (const auto e : v)
    //         std::cout << e << " ";
    //     std::cout << std::endl;
    // }

    auto clusters = makeClusters(l, dist);
    std::cout << "Clusters nr:" << clusters.size() << std::endl;
    for (const auto &e : clusters)
        std::cout << e.size() << "\t";

    // cout<<endl;

    // for(auto &v : clusters){
    //     for(auto &e:v){
    //         for(auto &w : e)
    //             cout<<w<<"\t";
    //         cout<<endl;
    //     }
    //     cout<<endl;
    // }

    std::cout << "\nFINISHED" << std::endl;
}