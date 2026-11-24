#include <metal_stdlib>
using namespace metal;
struct State { float4 positionAndMass; };
struct Tet { uint4 nodes; };
inline State femTrialNodeAtAlpha(State a,State b,float3 v,float alpha,float dt,bool update) {
    State result=a;result.positionAndMass.xyz+=alpha*dt*v;return result;
}
inline float3x3 registeredTrialF(thread const State* p,float3x3 inv,bool reference){
    return float3x3(p[1].positionAndMass.xyz-p[0].positionAndMass.xyz,
                    p[2].positionAndMass.xyz-p[0].positionAndMass.xyz,
                    p[3].positionAndMass.xyz-p[0].positionAndMass.xyz)*inv;
}
inline uint exercise(uint which,thread float4& record){
    State accepted[4],candidate[4],p[4];float4 direction[4];float3 next[4],beforeBalance[4];
    float w[4];Tet t={uint4(0,1,2,3)};float3x3 inv=float3x3(1.f);
    const float3 basis[4]={float3(0),float3(1,0,0),float3(0,1,0),float3(0,0,1)};
    float dt=1.f;bool reference=false;uint privateCount=which==2?3u:4u;
    const float lo=which==3?.61f:(which==1?.55f:.6f),hi=2.5f;
    for(uint k=0;k<4;++k){
        float mass=which==4 && k==3?16.f:1.f;
        candidate[k].positionAndMass=float4(basis[k],mass);accepted[k]=candidate[k];w[k]=1.f/mass;
        direction[k]=float4(k==1?-.3999996f:0.f,0,0,0);
        beforeBalance[k]=direction[k].xyz-float3(0x1p-22f,0,0);next[k]=beforeBalance[k];
    }
    float3 impulse=0.f;
    for(uint k=0;k<4;++k)impulse=fma(candidate[k].positionAndMass.w,next[k]-direction[k].xyz,impulse);
    next[0]-=impulse*w[0];
    for(uint k=0;k<4;++k)p[k]=femTrialNodeAtAlpha(accepted[k],candidate[k],next[k],1.f,dt,true);
    const float actual=determinant(registeredTrialF(p,inv,reference));record=float4(actual,NAN,NAN,0);
#include "represented_balance.inc"
    impulse=0;
    for(uint k=0;k<4;++k){
        impulse=fma(candidate[k].positionAndMass.w,next[k]-direction[k].xyz,impulse);
        p[k]=femTrialNodeAtAlpha(accepted[k],candidate[k],next[k],1.f,dt,true);
    }
    record.y=determinant(registeredTrialF(p,inv,reference));record.z=length(impulse);
    return usedTranslationBalance?9u:6u;
}
kernel void testBalance(device float4* out [[buffer(0)]],uint i [[thread_position_in_grid]]){
    if(i>=5)return;
    float4 r;uint status=exercise(i,r);r.w=float(status);out[i]=r;
}
