#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "NumiHumanRestingCommonField.metalinc"

kernel void solve_cardiac_targets(
    device const float4* volumeTrace [[buffer(0)]],
    constant MRHumanRestingCommonFieldGPU& parameters [[buffer(1)]],
    device const MRHumanRestingCommonCoordinateBoxGPU* boxes [[buffer(2)]],
    device MRHumanRestingCommonCoordinatesGPU* output [[buffer(3)]],
    uint step [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const auto volume=volumeTrace[step];
    const float target[7]={volume.x,volume.y,volume.z,volume.w,
        parameters.materialTargetVolumes.x,parameters.materialTargetVolumes.y,
        parameters.materialTargetVolumes.z};
    float coordinates[7]={},residual=INFINITY;
    uint iterations=0,matched=MR_INVALID_INDEX;
    const uint status=nmHumanRestingCommonSolveSimd(
        parameters,boxes,target,coordinates,lane,iterations,matched,residual);
    if(lane!=0u) return;
    // Keep exterior solutions for offline certification. This fixture has
    // no state transaction, acceptance path, or visual bindings.
    output[step].first=float4(coordinates[0],coordinates[1],coordinates[2],coordinates[3]);
    output[step].second=float4(coordinates[4],coordinates[5],coordinates[6],0);
    output[step].status=uint4(status,iterations,matched,0);
    output[step].diagnostics=float4(residual,0,0,0);
}
