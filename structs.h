#pragma once

#include <vector>

struct Layer {
	int input;
	int output;
};

struct WeightsData {
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<int> weights_offsets;
    std::vector<int> biases_offsets;
};