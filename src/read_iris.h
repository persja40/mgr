#pragma once
#include <fstream>
#include <vector>

#include <iostream>
#include <unistd.h>
#include <string>
#include <sstream>

std::vector<std::vector<float>> read_iris()
{
    std::vector<std::vector<float>> result{};
    std::ifstream fs("../data/iris.data", std::ios_base::in);
    for (std::string line; std::getline(fs, line);) //read stream line by line
    {
        std::istringstream in(line);
        float a, b, c, d;
        fs >> a >> b >> c >> d;
        std::vector<float> tmp{a, b, c, d};
        std::cout << a << b << c << d << std::endl;
        result.push_back(tmp);
    }
    return result;
}