#include "matter_kernels.metal"

namespace numi_matter_metal {
kernel void nm_contact_retained_edge_derivative_check(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    constexpr float shift = 0.01f;
    if (index != 0u) return;
    NMDeformableContactGPU contact = {};
    contact.identity.z = NM_DEFORMABLE_CONTACT_EDGE_EDGE;
    contact.barrier.z = input[16].x;
    contact.mollifier.w = input[16].y;
    MRCompensatedPositionGPU base[4], plus[4], minus[4];
    float3 direction[4];
    for (uint i = 0u; i < 4u; ++i) {
        base[i] = {input[2u*i], input[2u*i+1u]};
        const MRCompensatedPositionGPU end = {input[8u+2u*i], input[9u+2u*i]};
        const auto chord = mrCompensatedVectorAdd(end, mrCompensatedVectorNegate(base[i]));
        direction[i] = contactCompensatedPositionProjection(chord);
        plus[i] = mrCompensatedVectorAdd(base[i],
            mrCompensatedVectorScale(chord, {shift, 0.0f}));
        minus[i] = mrCompensatedVectorAdd(base[i],
            mrCompensatedVectorScale(chord, {-shift, 0.0f}));
    }
    float e0, ep, em, d0, dp, dm, slope;
    MRCompensatedScalar e0w, epw, emw;
    const bool v0 = deformableBarrierEnergyAt(contact, base, e0, d0, &e0w, direction, &slope);
    const bool vp = deformableBarrierEnergyAt(contact, plus, ep, dp, &epw);
    const bool vm = deformableBarrierEnergyAt(contact, minus, em, dm, &emw);
    const auto diff = mrCompensatedDivide(contactCompensatedSubtract(epw, emw),
        {2.0f*shift, 0.0f});
    output[0] = float4(slope, diff.high+diff.low, e0, d0);
    output[1] = float4(v0 && vp && vm ? 1.0f : 0.0f, dp, dm, 0.0f);
}
}
