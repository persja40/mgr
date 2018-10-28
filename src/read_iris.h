#pragma once
#include <fstream>
#include <vector>

#include <iostream>
#include <unistd.h>
#include <string>
#include <sstream>
#include <iterator>
#include <tuple>

std::string fn = "../data/iris.data";

std::vector<std::vector<float>> read_iris_cpu()
{
    std::vector<std::vector<float>> result{};
    std::ifstream fs(fn, std::ios_base::in);
    for (std::string line; std::getline(fs, line);) //read stream line by line
    {
        // std::cout << "line :" << line << std::endl;
        std::istringstream iss(line);
        std::vector<std::string> r((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        std::vector<float> tmp{std::stof(r[0]), std::stof(r[1]), std::stof(r[2]), std::stof(r[3])};
        // std::cout << std::stof(r[0]) << std::stof(r[1]) << std::stof(r[2]) << std::stof(r[3]) << std::endl;
        result.push_back(tmp);
    }
    return result;
}

std::tuple<std::vector<float>,int,int> read_iris_gpu()
{
    std::vector<float> result{};
    std::ifstream fs(fn, std::ios_base::in);
    int m = 0;
    for (std::string line; std::getline(fs, line);) //read stream line by line
    {
        // std::cout << "line :" << line << std::endl;
        std::istringstream iss(line);
        std::vector<std::string> r((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        // std::cout << std::stof(r[0]) << std::stof(r[1]) << std::stof(r[2]) << std::stof(r[3]) << std::endl;
        result.push_back(std::stof(r[0]));
        result.push_back(std::stof(r[1]));
        result.push_back(std::stof(r[2]));
        result.push_back(std::stof(r[3]));
        m++;
    }
    return std::make_tuple(result, m, 4);
}