#pragma once
#include <vector>
#include <list>
#include <cmath>

std::vector<std::vector<std::vector<float>>> makeClusters(std::list<std::vector<float>> &l, float dist)
{
    auto d_f = [](std::vector<float> &v1, std::vector<float> &v2) {
        float s = 0;
        for (int i = 0; i < v1.size(); i++)
            s += std::abs(v1[i] - v2[i]);
        return std::sqrt(s);
    };
    std::vector<std::vector<std::vector<float>>> clusters{};
    int cluster_nr = 0;
    while (!l.empty())
    {
        clusters.push_back(std::vector<std::vector<float>>());
        clusters[cluster_nr].push_back(*begin(l));
        l.erase(begin(l));
        for (int i = 0; i < clusters[cluster_nr].size(); i++)
            for (auto it = l.begin(); it != l.end();)
            {
                if (d_f(*it, clusters[cluster_nr][i]) <= dist)
                {
                    clusters[cluster_nr].push_back(*it);
                    it = l.erase(it);
                }
                else
                    it++;
            }
        cluster_nr++;
    };
    return clusters;
}