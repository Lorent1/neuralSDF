#include <perceptron/Perceptron.h>

#include <vector>
#include <iostream>
#include <chrono>
#include <Perceptron/mat_functions.h>
#include <omp.h>

#define K 1

struct prop_data {
	Layer* layers;
	WeightsData data;

	int layersNum;
};

struct output_data {
	float* t;
	float* h;
	int* offsets;
	float distance;
};

output_data Forward_propagation(float3 coords, prop_data* props) {
	matrix result = init_mat(coords, 3, 1);

	Layer* layers = props->layers;
	WeightsData data = props->data;

	int layersNum = props->layersNum;

	// fill per layer struct
	output_data out;
	out.offsets = props->data.biases_offsets.data();
	out.t = (float*)calloc(props->data.biases.size(), sizeof(float));
	out.h = (float*)calloc(props->data.biases.size(), sizeof(float));

	for (int i = 0; i < layersNum - 1; i++) {
		matrix weights = { data.weights.data(), layers[i].output, layers[i].input};
		result = multiply_mat(&weights, &result, data.weights_offsets[i]);

		matrix biases = { data.biases.data(), layers[i].output, 1 };
		result = add_mat(&result, &biases, data.biases_offsets[i]);

		memcpy(&out.t[data.biases_offsets[i]], result.data, layers[i].output * sizeof(float));

		sin_mat(&result, layers[i].output, 1);
		memcpy(&out.h[data.biases_offsets[i]], result.data, layers[i].output * sizeof(float));
	}

	// doesn't apply sin func to the output layer
	matrix weights = { data.weights.data(), layers[layersNum - 1].output, layers[layersNum - 1].input };
	result = multiply_mat(&weights, &result, data.weights_offsets[layersNum - 1]);

	matrix biases = { data.biases.data(), layers[layersNum - 1].output, 1 };
	result = add_mat(&result, &biases, data.biases_offsets[layersNum - 1]);

	// the final result is matrix 1x1 - number
	out.distance = result.data[0];

	return out;
}

float3 EstimateNormal(float3 z, float eps, prop_data* data) {
	float3 z1 = z + float3(eps, 0, 0);
	float3 z2 = z - float3(eps, 0, 0);
	float3 z3 = z + float3(0, eps, 0);
	float3 z4 = z - float3(0, eps, 0);
	float3 z5 = z + float3(0, 0, eps);
	float3 z6 = z - float3(0, 0, eps);
	float dx = Forward_propagation(z1, data).distance - Forward_propagation(z2, data).distance;
	float dy = Forward_propagation(z3, data).distance - Forward_propagation(z4, data).distance;
	float dz = Forward_propagation(z5, data).distance - Forward_propagation(z6, data).distance;
	return normalize(float3(dx, dy, dz));
}


float3 getLight(float3 pos, float3 rd, float3 color, prop_data* data) {
	float3 lightPos = float3(2.0f, 20.0f, -5.0f);
	float3 L = normalize(lightPos - pos);
	float3 N = EstimateNormal(pos, 1e-3, data);
	float3 V = -1 * rd;
	float3 R = reflect(-1 * L, N);

	float3 specColor = float3(0.5f);
	float3 specular = 1.3f * specColor * pow(clamp(dot(R, V), 0.0f, 1.0f), 10.0f);
	float3 ambient = 0.05f * color;
	float3 fresnel = 0.15f * color * pow(1.0f + dot(rd, N), 3.0f);

	float3 dif = color * clamp(dot(N, L), 0.1f, 1.0f);

	return dif;
}

static inline uint32_t RealColorToUint32(float4 real_color) {
	int red = std::max(0, std::min(255, (int)(real_color[0] * 255.0f)));
	int green = std::max(0, std::min(255, (int)(real_color[1] * 255.0f)));
	int blue = std::max(0, std::min(255, (int)(real_color[2] * 255.0f)));
	int alpha = std::max(0, std::min(255, (int)(real_color[3] * 255.0f)));

	return red | (green << 8) | (blue << 16) | (alpha << 24);
}

float2 getUV(float2 offset, int x, int y, int width, int heigth) {
	float ratio = (float)width / heigth;
	return ((float2(x, y) - offset) * 2.0f / float2(width, heigth) - 1.0f) * float2(ratio, 1.0f);
}

float MSE(float act, float pred) {
	float dif = act - pred;
	dif *= dif;

	return dif;
}

float MSE_grad(float act, float pred) {
	float dif = act - pred;
	return dif * 2;
}

float* to_full(float t, int size) {
	float* full = (float*)calloc(size, sizeof(float));
	for (int i = 0; i < size; i++) {
		full[i] = size;
	}
	return full;
}

void Perceptron::Learn(float3* points, float* expected){
	//prop_data props = { layers, data, layersNum };
	//output_data out = Forward_propagation(points[0], &props);
	//float act = out.distance;

	//float t = MSE(act, expected[0]);
	//float* t_full = to_full(t, layers[layersNum - 1].output);

	//float* dE_dW = multiply_mat(out.h, );

}


void Perceptron::kernel2D_Render(uint32_t* out_color, uint32_t width, uint32_t height){
	const float MAX_DISTANCE = 2.0f;
	prop_data props = { layers, data, layersNum };

	#pragma omp for
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float3 pixel = float3(0.0f);
			float3 ro = float3(0.0f, 0.4f, -0.8f);
			float2 uv = getUV(float2(0.0f), x, y, width, height);
			float3 rd = normalize(float3(uv.x, uv.y, 1.0f));

			float t = 0.0f;
			float3 p;

			for (int i = 0; i < 500; i++) {
				p = ro + rd * t;

				float distance = Forward_propagation(p, &props).distance;

				t += distance;

				if (t > MAX_DISTANCE || distance < 1e-5) break;
			}

			if (t < MAX_DISTANCE) {
				pixel = getLight(p, rd, float3(1.0f), &props);
			}

			out_color[y * width + x] = RealColorToUint32(float4(pixel.x, pixel.y, pixel.z, 1.0f));
		}
		std::cout << y << " ";
	}
}

void Perceptron::RayMarch(uint32_t* out_color, uint32_t width, uint32_t height) {
	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel
	{
		kernel2D_Render(out_color, width, height);
	}

	totalTime = float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()) / 1000.f;
};

void Perceptron::Test(float3* points, float* expected_distances, uint32_t size) {
	prop_data props = { layers, data, layersNum };

	for (int i = 0; i < size; i++) {
		float res = Forward_propagation(points[i], &props).distance - expected_distances[i];
		if (res > 1e-6) {
			std::cout << "Deviation more than eps";
		}
	}
}

void Perceptron::GetExecutionTime(const char* a_funcName, float a_out[4]) {
	if (std::string(a_funcName) == "RayMarch")
		a_out[0] = totalTime;
}