#include "src/matter_kernels.metal"
using namespace numi_matter_metal;
kernel void test_registered_plane_floor(device const float4* input [[buffer(0)]],
                                        device float* result [[buffer(1)]],
                                        uint i [[thread_position_in_grid]]) {
    const float4 p = input[i];
    result[i] = registeredPlaneCertificateFloor(float3(0, 0, 1), p.xyz, p.w, 1.0e-5f, 1.0e-9f);
}
