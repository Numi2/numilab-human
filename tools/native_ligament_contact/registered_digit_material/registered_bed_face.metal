#include "src/matter_kernels.metal"

// Integrate the stationary fixed-plane barrier on one registered reference face.
// Lanes 0..15 and 16..31 integrate the two possible pieces of its clipped triangle.
struct BedFace {
    float4 gapsAndArea;
    float4 materialDensity;
};
struct BedFaceResponse {
    float4 normalForce;
    float4 stiffnessFirstRow;
    float4 stiffnessLowerBlock;
};
kernel void registered_bed_face(device const BedFace* faces [[buffer(0)]],
                                device BedFaceResponse* response [[buffer(1)]],
                                constant float2& stepAndWidth [[buffer(2)]],
                                uint face [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    using namespace numi_matter_metal;
    const BedFace input=faces[face];
    const float3 gap=input.gapsAndArea.xyz;
    const float dt=stepAndWidth.x, width=stepAndWidth.y;
    if (min3(gap.x,gap.y,gap.z)>=width) {
        if (lane==0) response[face]=BedFaceResponse{};
        return;
    }
    constexpr float3 corner[3]={float3(1,0,0),float3(0,1,0),float3(0,0,1)};
    float3 clip[4];
    uint count=0;
    for (uint i=0;i<3;++i) {
        const uint j=(i+1)%3;
        const bool inside=gap[i]<width;
        if (inside) clip[count++]=corner[i];
        if (inside!=(gap[j]<width))
            clip[count++]=mix(corner[i],corner[j],(width-gap[i])/(gap[j]-gap[i]));
    }
    float3 force=0, first=0, lower=0;
    const uint piece=lane/16;
    if (piece+2<count) {
        // Four-point Gauss-Legendre data on [0,1]; Duffy maps the square to a triangle.
        constexpr float points[4]={.0694318442f,.3300094782f,.6699905218f,.9305681558f};
        constexpr float weights[4]={.1739274226f,.3260725774f,.3260725774f,.1739274226f};
        const uint i=(lane%16)/4,j=lane%4;
        const float u=points[i],v=(1-u)*points[j];
        const float3 a=clip[0],b=clip[piece+1],c=clip[piece+2];
        const float3 basis=(1-u-v)*a+u*b+v*c;
        const float area=input.gapsAndArea.w*abs(dot(a,cross(b,c)));
        const float measure=2*(1-u)*weights[i]*weights[j]*area*
                            dot(input.materialDensity.xyz,basis);
        const float4 ipc=ipcBarrierResponse(dot(basis,gap),width,1.f,dt,0.f,false,0.f,0.f);
        force=(measure*(ipc.x/dt))*basis;
        const float k=measure*(ipc.y/(dt*dt));
        first=k*basis.x*basis;
        lower=k*float3(basis.y*basis.y,basis.y*basis.z,basis.z*basis.z);
    }
    force=simd_sum(force);first=simd_sum(first);lower=simd_sum(lower);
    if (lane==0) response[face]={float4(force,0),float4(first,0),float4(lower,0)};
}
