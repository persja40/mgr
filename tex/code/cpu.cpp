#include <thread>
#include <future>
#include <vector>
using namespace std;
(...)
vector<float> data;
(...)
float v = 0;
unsigned nr_threads = thread::hardware_concurrency();
auto reduce = [&](int thread_nr) -> float {
    float result = 0;
    for (int i = thread_nr; i < data.size(); i += nr_threads)
        result += data[i];
    return result;
};
vector<future<float>> fut_handler{};
for (int i = 0; i < nr_threads; i++)
    fut_handler.push_back(std::async(launch::async, reduce, i));
for (auto &e : fut_handler)
    v += e.get();