#include "gaussian.h"

#include <cmath>
#include <random>
#include <vector>

namespace dkvr
{

    namespace
    {
        std::random_device rd{};
        std::mt19937 gen{ rd() };
        std::normal_distribution<float> nd{ 0.0f, 1.0f };

        // unscaled gaussian function
        static float GaussFunction(int x, int size)
        {
            float mean = size / 2.0f;
            float sigma = size / 6.0f;
            float dividend = x - mean;
            return expf(-dividend * dividend / (2.0f * sigma * sigma));
        }

    }

    float Gaussian::NormalDistribution()
    {
        return nd(gen);
    }

    std::vector<float> Gaussian::GenerateGaussArray(int sampling_rate, float time, bool normalize)
    {
        int size = static_cast<int>(sampling_rate * time);
        std::vector<float> result;
        result.resize(size);

        // generate gauss
        for (int i = 0; i < size; i++)
            result[i] = GaussFunction(i, size);

        // normalize and scale
        if (normalize)
        {
            float sum = 0;
            for (float f : result)
                sum += f;

            for (float& f : result)
                f /= sum;
        }
        
        return result;
    }

}   // namespace dkvr