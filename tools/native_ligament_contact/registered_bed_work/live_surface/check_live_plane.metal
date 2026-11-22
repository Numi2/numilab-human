#include "src/matter_kernels.metal"
using namespace numi_matter_metal;
kernel void make_bed_test_state(device const float4* input [[buffer(0)]],
                                device NMFEMNodeStateGPU* nodes [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    NMFEMNodeStateGPU node={};
    node.positionAndMass=float4(0,0,input[i].x,input[i].y);
    node.velocityAndInverseMass=float4(0,0,input[i].z,1.f/input[i].y);
    nodes[i]=node;
}
kernel void test_live_bed_rows(constant NMPrimalContactArguments& contact [[buffer(0)]],
                               device const NMFEMNodeStateGPU* nodes [[buffer(1)]],
                               device const NMFEMNodeStateGPU* accepted [[buffer(2)]],
                               device const float4* direction [[buffer(3)]],
                               device float4* records [[buffer(4)]],
                               constant float2& params [[buffer(5)]],
                               uint i [[thread_position_in_grid]]) {
    const float3 row=registeredPlaneRow(contact,i,params.y,params.x,nodes,0.f,accepted,true);
    const float neighbor=planeNeighborAction(contact,i,direction).z;
    records[i]=float4(row.x,row.y*direction[i].z+neighbor,row.y,neighbor);
}
