#pragma once
#include <fstream>
#include <vector>

#include <iostream>
#include <unistd.h>
#include <string>

std::vector<std::vector<float>> read_iris()
{
    std::vector<std::vector<float>> result{};
    std::fstream fs("../data/iris.data", std::ios::in);
    char cwd[2000];
    std::cout << getcwd(cwd, sizeof(cwd)) << std::endl;
    while (!fs.eof())
    {
        float a, b, c, d;
        std::string trash;
        fs >> a >> b >> c >> d >> trash;
        std::vector<float> tmp{a, b, c, d};
        result.push_back(tmp);
    }
    return result;
}