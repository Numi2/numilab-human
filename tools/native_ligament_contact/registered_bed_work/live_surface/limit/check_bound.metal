#include "src/matter_kernels.metal"
using namespace numi_matter_metal;
kernel void test_registered_plane_fraction(device const float4* cases [[buffer(0)]],device float* result [[buffer(1)]],uint i [[thread_position_in_grid]]){
    const float4 c=cases[i];result[i]=registeredPlaneFraction(c.x,c.y,c.z,c.w);
}
