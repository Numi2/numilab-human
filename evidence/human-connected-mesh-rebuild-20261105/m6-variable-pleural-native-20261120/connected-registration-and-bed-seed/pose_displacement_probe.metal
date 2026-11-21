#include <metal_stdlib>
using namespace metal;
#include "numi/matter/fem_compensated_deformation.h"

kernel void check_pose(
    device const float* displacement [[buffer(0)]],
    device const uint4* corners [[buffer(1)]],
    device const float* inverses [[buffer(2)]],
    device float* result [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= count) return;
    float high[12], low[12], inverse[9], outHigh[9], outLow[9];
    const float identity[9] = {1,0,0,0,1,0,0,0,1};
    for (uint i = 0; i < 4; ++i) {
        const uint node = corners[row][i];
        for (uint axis = 0; axis < 3; ++axis) {
            high[3*i+axis] = displacement[6*node+axis];
            low[3*i+axis] = displacement[6*node+3+axis];
        }
    }
    for (uint i = 0; i < 9; ++i) inverse[i] = inverses[9*row+i];
    nmFEMReferenceDeformationWords(high, low, inverse, identity, outHigh, outLow);
    for (uint i = 0; i < 9; ++i) {
        result[18*row+i] = outHigh[i];
        result[18*row+9+i] = outLow[i];
    }
}
