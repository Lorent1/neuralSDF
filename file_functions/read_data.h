#pragma once
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

#include <LiteMath.h>

using json = nlohmann::json;
using namespace LiteMath;

class Files{
public:
    static void parse_layers(std::string path, std::vector<Layer>* layers){
        std::ifstream f(path);
        json data = json::parse(f);

        try {
            if (data.contains("inputLayer")) {
                layers->push_back(read_layer(data["inputLayer"]));
            }
            else {
                throw "No input layer!";
            }

            if (data.contains("hiddenLayers")) {
                int n = data["hiddenLayers"].size();

                for (int i = 0; i < n; i++) {
                    layers->push_back(read_layer(data["hiddenLayers"][i]));
                }
            }


            if (data.contains("outputLayer")) {
                layers->push_back(read_layer(data["outputLayer"]));
            }
            else {
                throw "No output layer!";
            }

            check_layers(*layers);
        }
        catch (const char* e) {
            std::cout << e;
            exit(1);
        }
    }

    static void parse_weights(const char* path, std::vector<Layer> layers, WeightsData* data) {
        FILE* file = fopen(path, "rb");

        if (file == nullptr) { std::cout << "Impossible to read weight file"; exit(1); }

        // compute weights

        int n = layers.size();
        int weights_size = 0;
        int biases_size = 0;

        data->weights_offsets.resize(n);
        data->biases_offsets.resize(n);
        
        for (int i = 0; i < n; i++) {
            data->weights_offsets[i] = weights_size;
            data->biases_offsets[i] = biases_size;
            weights_size += layers[i].input * layers[i].output;
            biases_size += layers[i].output;
        }

        data->weights.resize(weights_size);
        data->biases.resize(biases_size);
        
        for (int i = 0; i < n; i++) {
            int s = layers[i].input * layers[i].output;

            for (int j = 0; j < s; j++) {
                float num;
                fread(&num, sizeof(float), 1, file);
                data->weights[j + data->weights_offsets[i]] = num;
            }
            for (int j = 0; j < layers[i].output; j++) {
                float num;
                fread(&num, sizeof(float), 1, file);
                data->biases[j + data->biases_offsets[i]] = num;
            }
        }
    }

    static void parse_points(const char* path, std::vector<float3>* points, std::vector<float>* expected) {
        FILE* file = fopen(path, "rb");
        if (file == nullptr) { std::cout << "Impossible to read weight file"; exit(1); }

        int n;
        fread(&n, sizeof(int), 1, file);
        points->resize(n);
        expected->resize(n);

        for (int i = 0; i < n; i++) {
            float3 point;
            fread(&point, sizeof(float3), 1, file);
            (*points)[i] = point;
        }

        for (int i = 0; i < n; i++) {
            float distance;
            fread(&distance, sizeof(float), 1, file);
            (*expected)[i] = distance;
        }
    }

private:
    static Layer parse_layer(json data) {
        if (data.contains("input") && data.contains("output")) {
            Layer layer = { data["input"], data["output"] };
            return layer;
        }
        else {
            throw "No input or output info in layer!";
        }
    }

    static void check_layers(std::vector<Layer> layers) {
        for (int i = 0; i < layers.size() - 1; i++) {
            if (layers[i].output != layers[i + 1].input) {
                throw "Invalid input-output data between layers";
            }
        }
    }
};
