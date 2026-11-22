#include "src/matter_kernels.metal"
using namespace numi_matter_metal;
kernel void measure_bed_projection(device const float4* input [[buffer(0)]],
                                   device float4* output [[buffer(1)]],
                                   uint i [[thread_position_in_grid]]) {
    const uint base=i*5;
    const float4 b=input[base], v=input[base+1], a=input[base+2], dir=input[base+3], mu=input[base+4];
    const BedProjection p=projectBedMomentum(b.xyz,v.xyz,v.w,b.w,dir.w,mu.xy);
    output[i*4]=float4(p.impulse,p.tangent.z);
    output[i*4+1]=float4(b.xyz+p.impulse,0);
    output[i*4+3]=float4(admissibleBedDirection(dir.xyz,v.xyz,b.w,dir.w,p.tangent.z),0);
    output[i*4+2]=float4(applyBedMomentum(a.xyz,dir.xyz,v.w,p.tangent),0);
}
