#include <metal_stdlib>
using namespace metal;
#include "metalrobo/compensated_translation_gpu.h"
kernel void checkCertifiedEdges(device const float* rows [[buffer(0)]],
                                device uint* failures [[buffer(1)]],
                                uint id [[thread_position_in_grid]]) {
    device const float* a=rows+5*id;
    auto d=mrCompensatedAdd({a[0],a[1]},{-a[2],-a[3]});
    failures[id]=uint(d.high!=a[4] || d.low!=0.0f);
}
