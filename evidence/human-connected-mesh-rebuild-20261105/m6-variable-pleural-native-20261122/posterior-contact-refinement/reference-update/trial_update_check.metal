#include "src/matter_kernels.metal"

kernel void check_material_update(device float4* output [[buffer(0)]],
                                  uint i [[thread_position_in_grid]]) {
    using namespace numi_matter_metal;
    if (i>=7u) return;
    constexpr float dt=0x1p-16f;
    constexpr float alpha=0x1p-17f;
    NMFEMNodeStateGPU accepted={};
    accepted.restAndFixed=float4(0.133f,0.f,0.f,i==3u?1.f:0.f);
    accepted.referenceDisplacementAndMode=float4(0.01234567f,0.f,0.f,i==5u?0.f:1.f);
    accepted.referenceDisplacementLow=float4(-0x1p-35f,0.f,0.f,0.f);
    accepted.velocityAndInverseMass=float4(0.75f,0.f,0.f,1.f);
    NMFEMNodeStateGPU next=accepted;
    const auto u=mrCompensatedAdd({accepted.referenceDisplacementAndMode.x,accepted.referenceDisplacementLow.x},
                                  {dt*0.75f,0.f});
    next.referenceDisplacementAndMode.x=u.high;
    next.referenceDisplacementLow.x=u.low;
    const auto f=femTrialNodeAtAlpha(accepted,next,float3(0.0012f,0.f,0.f),i==2u?0.f:alpha,dt,i!=4u);
    const auto g=(i==1u || i==6u) ? femTrialNodeAtAlpha(accepted,f,
        float3(i==6u?-0.0012f:0.0012f,0.f,0.f),alpha,dt,true) : f;
    const auto du=mrCompensatedAdd({g.referenceDisplacementAndMode.x,g.referenceDisplacementLow.x},
                                    {-u.high,-u.low});
    output[i]=float4(du.high,du.low,g.velocityAndInverseMass.x,g.deltaVelocity.x);
}
