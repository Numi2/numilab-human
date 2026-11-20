#pragma once

#include "numi/matter/numi_human_shared.h"

namespace numi::matter {

struct HumanGlideRefreshTrial {
    NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
    id<MTLComputePipelineState> refresh = nil;
    id<MTLComputePipelineState> finish = nil;
    id<MTLBuffer> samples = nil;
    id<MTLBuffer> poses = nil;
    id<MTLBuffer> previousForce = nil;
    id<MTLBuffer> rowImpulses = nil;
    id<MTLBuffer> wrenches = nil;
    id<MTLBuffer> external = nil;
};

// Private bridge for the paired experiment. The same host thread enters
// encodeWithGravityScale synchronously with the strictly scoped plan.
inline thread_local const HumanGlideRefreshTrial* humanGlideRefresh = nullptr;

} // namespace numi::matter
