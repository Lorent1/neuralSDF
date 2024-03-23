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

#define LEARN

using namespace LiteMath;


int main(){
	WeightsData data;
	std::vector<Layer> layers;

	Files::parse_layers("sdf_files/layers2.json", &layers);
	Files::parse_weights("sdf_files/sdf2_weights.bin", layers, &data);

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
	std::vector<float> distances;
	std::vector<float3> coords;

	Files::parse_points("sdf_files/sdf2_points.bin", &coords, &distances);

	pImpl->Learn();
#endif
#ifdef TEST
	std::vector<float> test_distances;
	std::vector<float3> test_coords;
	Files::parse_coords("sdf_files/sdf2_test.bin", &test_coords, &test_distances);
	pImpl->Test(test_coords.data(), test_distances.data(), test_distances.size());
#endif
	pImpl->RayMarch(pixelData.data(), WIDTH, HEIGHT);
	//std::cout << "here";

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
