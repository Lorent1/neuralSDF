#pragma once

#include <structs.h>
#include <vector>
#include <LiteMath.h>

using namespace LiteMath;

class Perceptron{
public:
    Perceptron() {};

    void setStartData(Layer* l, WeightsData d, int ln){
        layers = l;
        data = d;
        layersNum = ln;
    }

    virtual void Learn();
    virtual void kernel2D_Render(uint32_t* out_color, uint32_t width, uint32_t height);
    virtual void Test(float3* coords, float* distances, uint32_t size);
    virtual void RayMarch(uint32_t* out_color [[size("width*height")]], uint32_t width, uint32_t height);

    virtual void CommitDeviceData() {}                                // will be overriden in generated class
    virtual void UpdateMembersPlainData() {}                              // will be overriden in generated class (optional function)
	virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);   // will be overriden in generated class
protected:
    Layer* layers;
    WeightsData data;
    int layersNum;

    float totalTime;
};

