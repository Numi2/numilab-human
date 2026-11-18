#include <metal_stdlib>
using namespace metal;
struct Point {
    packed_float3 base;
    packed_float3 displacement[7];
};
kernel void pose_common_field(
    device const Point* points [[buffer(0)]],
    constant float* coordinate [[buffer(1)]],
    device packed_float3* positions [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    float3 local=points[row].base;
    for(uint c=0;c<7;++c) local+=float3(points[row].displacement[c])*coordinate[c];
    positions[row]=local;
}
