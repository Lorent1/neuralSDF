#include <structs.h>
#include "LiteMath.h"
#include "file_functions/read_data.h"
#include <memory>
#include <iomanip>
#include <sstream>
#include "perceptron/Perceptron.h"

#include "Image2d.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<RayMarcher> CreateRayMarcher_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif

//#define LEARN
//#define TEST

using namespace LiteMath;

struct FilePaths {
	std::string layers_path;
	std::string weights_path;
	std::string points_path;
	std::string test_points_path;
};

FilePaths setPaths(std::string filename) {
	std::string layers_path = "sdf_files/layers" + filename + ".json";
	std::string weights_path = "sdf_files/sdf" + filename + "_weights.bin";
	std::string points_path = "sdf_files/sdf" + filename + "_points.bin";
	std::string test_points_path = "sdf_files/sdf" + filename + "_test.bin";

	return { layers_path, weights_path, points_path, test_points_path };
}

int main(){
	WeightsData data;
	std::vector<Layer> layers;
	FilePaths paths = setPaths("1");

	Files::parse_layers(paths.layers_path.c_str(), &layers);
	Files::parse_weights(paths.weights_path.c_str(), layers, &data);

	uint32_t WIDTH = 256;
	uint32_t HEIGHT = 256;

	std::vector<uint> pixelData(WIDTH * HEIGHT);

	std::shared_ptr<Perceptron> pImpl = nullptr;

#ifdef USE_VULKAN
	bool onGPU = true; // TODO: you can read it from command line
	if (onGPU)
	{
		auto ctx = vk_utils::globalContextGet(enableValidationLayers, 0);
		pImpl = CreateRayMarcher_Generated(ctx, WIDTH * HEIGHT);
	}
	else
#else
	bool onGPU = false;
#endif

	pImpl = std::make_shared<Perceptron>();
	pImpl->setStartData(layers.data(), data, layers.size());

#ifdef LEARN
	std::vector<float> expected_distances;
	std::vector<float3> points;

	Files::parse_points(paths.points_path.c_str(), &points, &expected_distances);

	pImpl->Learn(points.data(), expected_distances.data());
#endif
#ifdef TEST
	std::vector<float> test_distances;
	std::vector<float3> test_points;

	Files::parse_points(paths.test_points_path.c_str(), &test_points, &test_distances);
	pImpl->Test(test_points.data(), test_distances.data(), test_distances.size());
#else
	pImpl->RayMarch(pixelData.data(), WIDTH, HEIGHT);
#endif

	float timings[4] = { 0,0,0,0 };
	pImpl->GetExecutionTime("RayMarch", timings);

	std::stringstream strOut;

	if (onGPU)
		strOut << std::fixed << std::setprecision(2) << "images/out_gpu_" << WIDTH << "x" << HEIGHT << ".bmp";
	else
		strOut << std::fixed << std::setprecision(2) << "images/out_cpu_" << WIDTH << "x" << HEIGHT << ".bmp";

	std::string fileName = strOut.str();

	LiteImage::SaveBMP(fileName.c_str(), pixelData.data(), WIDTH, HEIGHT);

	std::cout << "timeRender = " << timings[0] << " ms, timeCopy = " << timings[1] + timings[2] << " ms " << std::endl;

	pImpl = nullptr;

	return 0;
}
