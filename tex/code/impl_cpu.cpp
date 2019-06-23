float g_h(vector<vector<float>> &data, float h, int kth)
{
    int m = data.size();
    int n = 1;
    unsigned nr_threads = thread::hardware_concurrency();
    float answer = 0;
    auto par_fun = [&](int nr) -> float {
        float r = 0;
        float xTx = 0;
        for (int i = nr; i < data.size(); i += nr_threads)
            for (int j = 0; j < m; j++)
            {
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

    float si_avg = accumulate(begin(answer), end(answer), 0.0, [&](float r, float x) {
        return r + log(x);
    });
    si_avg = expf(si_avg / m);

    //now si
    for (auto &e : answer)
        e = pow(e / si_avg, -c);

    return answer;
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