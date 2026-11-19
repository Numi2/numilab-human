#include <metal_stdlib>
using namespace metal;

struct Point {
    packed_float3 base;
    packed_float3 displacement[7];
};
struct LocalExpansion {
    packed_float3 high;
    packed_float3 low;
};

static inline float3 errorFreeAdd(float3 a, float3 b, thread float3& residual) {
    const float3 sum = a + b;
    const float3 fromSecond = sum - a;
    residual = (a - (sum - fromSecond)) + (b - fromSecond);
    return sum;
}

static inline void addBasis(thread float3& high, thread float3& low, float3 d, float q) {
    const float3 product = d * q;
    const float3 productLow = fma(d, float3(q), -product);
    float3 roundoff;
    const float3 sum = errorFreeAdd(high, product, roundoff);
    const float3 correction = low + (roundoff + productLow);
    high = errorFreeAdd(sum, correction, low);
}

kernel void pose_common_field(
    device const Point* points [[buffer(0)]],
    constant float* coordinate [[buffer(1)]],
    device LocalExpansion* positions [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    float3 high = points[row].base;
    float3 low = 0;
    for (uint c=0;c<7;++c) addBasis(high, low, points[row].displacement[c], coordinate[c]);
    positions[row] = {packed_float3(high), packed_float3(low)};
}
