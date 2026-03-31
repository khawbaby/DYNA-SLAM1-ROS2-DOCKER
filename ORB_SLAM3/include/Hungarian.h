#pragma once
#include <vector>
#include <limits>
#include <algorithm>

namespace ORB_SLAM3
{
    class Hungarian
    {
    public:
        std::vector<int> solve(const std::vector<std::vector<float>>& cost);
    };
}