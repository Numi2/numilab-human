#include <metal_stdlib>
using namespace metal;
#include "pelvic_crossing_predicates.metalinc"
kernel void testGeometrySigns(device const float4* p [[buffer(0)]],device int4* output [[buffer(1)]],
    uint i [[thread_position_in_grid]]){
    float3 a[3],b[3];
    for(uint j=0;j<3;++j){a[j]=p[6*i+j].xyz;b[j]=p[6*i+j+3].xyz;}
    output[i]=int4(pelvicOrient(a[0],a[1],a[2],b[0]),pelvicExactOrient(a[0],a[1],a[2],b[0]),pelvicCrosses(a,b),0);
}
