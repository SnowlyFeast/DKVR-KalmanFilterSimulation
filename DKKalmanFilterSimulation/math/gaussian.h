#pragma once

#include <cmath>
#include <vector>

namespace dkvr
{
    class Gaussian
    {
    public:
        static float NormalDistribution();

        // sampling_rate: Hz, time: second
        static std::vector<float> GenerateGaussArray(int sampling_rate, float time, bool normalize = true);
    };
}